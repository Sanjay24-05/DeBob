import { existsSync, readFileSync } from 'node:fs'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { join } from 'node:path'
import type { Edge, Node } from '../graph/types.js'
import {
  openDb,
  readManifest,
  SqlitePersistenceAdapter,
  type Manifest,
} from '../persistence/sqlite.js'

/** One influential-signal entry from a prediction row's `risk_reasons`. */
interface RiskReason {
  feature: string
  label: string
  value: number
  importance: number
}

type VisualiserNode = Pick<
  Node,
  'id' | 'type' | 'name' | 'filePath' | 'layer' | 'startLine' | 'endLine' | 'confidence' | 'dataSource' | 'metadata'
> & {
  /** LLM-written module summary, joined from semantic_enrichments. */
  responsibility?: string
  /** Model that produced `responsibility`, for attribution in the UI. */
  responsibilityModel?: string
  /** Model-assigned risk score, joined from ml/artifacts/predictions.json. */
  riskProbability?: number
  /** Whether the model classed this file risky (probability >= 0.5). */
  predictedRisky?: boolean
  /** Influential signals behind the score — NOT causal explanations. */
  riskReasons?: RiskReason[]
}
type VisualiserEdge = Pick<Edge, 'id' | 'source' | 'target' | 'type' | 'confidence' | 'dataSource'>

/**
 * Coverage and provenance for the ML overlay.
 *
 * `available: false` carries a `hint` instead of counts — the overlay is a strictly
 * optional layer, because `ml/artifacts/` is gitignored and a fresh clone ships the
 * graph without any predictions.
 */
interface PredictionInfo {
  available: boolean
  /** Why the overlay is unavailable, phrased as the command that would fix it. */
  hint?: string
  /** Rows actually present in the file — `--top` truncates what gets written. */
  rowsShown?: number
  /** Candidates the model scored, from the sidecar. Usually far larger than `rowsShown`. */
  candidateCount?: number
  modelName?: string
  /** True when the predictions were scored against a different graph than the one loaded. */
  stale?: boolean
}

interface VisualiserPayload {
  manifest: Manifest | null
  nodes: VisualiserNode[]
  edges: VisualiserEdge[]
  predictionInfo: PredictionInfo
}

/**
 * Reads the persisted graph once, then serves it through a local read-only
 * HTTP server for the inline visualisation.
 */
export async function startVisualiserServer(
  repoRoot: string,
  options: { port?: number } = {},
): Promise<{ url: string; close: () => void }> {
  const contextDbPath = join(repoRoot, '.debob', 'context.db')
  if (!existsSync(contextDbPath)) {
    throw new Error("No graph found. Run 'debob init' first.")
  }

  const { db, dbPath } = await openDb(repoRoot)
  const adapter = new SqlitePersistenceAdapter(db, dbPath)
  let graph
  let enrichments
  try {
    graph = adapter.readGraph()
    // `Node.responsibility` is never populated in the persisted graph — LLM output lives
    // only in semantic_enrichments — so it has to be joined on read, the same way
    // src/engine/explain.ts does it. Without this the visualiser shows nothing watsonx wrote.
    enrichments = adapter.readSemanticEnrichments()
  } finally {
    // sql.js allocates the database in memory, including for this read-only use.
    adapter.close()
  }

  const responsibilities = new Map<string, { value: string; modelId: string }>()
  for (const enrichment of enrichments) {
    if (enrichment.field !== 'responsibility') continue
    responsibilities.set(enrichment.nodeId, {
      value: enrichment.value,
      modelId: enrichment.modelId,
    })
  }

  const manifest = readManifest(repoRoot)
  // Predictions are read once here, alongside the graph, and the payload is frozen for the
  // lifetime of the server. Re-running `debob predict-risk` therefore needs a `visualise`
  // restart to show up — that is the same contract the graph itself has, not a bug.
  const { info: predictionInfo, rows: predictions } = readPredictions(repoRoot, manifest)

  const payload: VisualiserPayload = {
    manifest,
    nodes: Array.from(graph.nodes.values()).map(node =>
      serialiseNode(node, responsibilities.get(node.id), predictions.get(node.id)),
    ),
    edges: graph.edges.map(serialiseEdge),
    predictionInfo,
  }
  const requestedPort = options.port ?? 7842
  if (!Number.isInteger(requestedPort) || requestedPort < 1 || requestedPort > 65535) {
    throw new Error('Port must be an integer between 1 and 65535.')
  }

  for (let offset = 0; offset < 5 && requestedPort + offset <= 65535; offset += 1) {
    const port = requestedPort + offset
    try {
      const server = await listen(port, payload)
      return {
        url: 'http://localhost:' + port,
        close: () => server.close(),
      }
    } catch (error) {
      if (isAddressInUse(error)) continue
      throw error
    }
  }

  throw new Error('Unable to start graph visualiser: the next 5 ports are already in use.')
}

/** One row as written by ml/predict_risk.py. */
interface PredictionRow {
  riskProbability: number
  predictedRisky: boolean
  riskReasons: RiskReason[]
}

/**
 * Loads the ML risk predictions, if the pipeline has been run.
 *
 * Returns an empty map and an unavailable `PredictionInfo` for every failure mode rather
 * than throwing: the overlay is optional, and a missing or malformed prediction file must
 * never stop the graph from rendering.
 */
function readPredictions(
  repoRoot: string,
  manifest: Manifest | null,
): { info: PredictionInfo; rows: Map<string, PredictionRow> } {
  const rows = new Map<string, PredictionRow>()
  const jsonPath = join(repoRoot, 'ml', 'artifacts', 'predictions.json')

  if (!existsSync(jsonPath)) {
    // `debob predict-risk` defaults to --output ...predictions.csv, so the CSV existing
    // alone is the likeliest miss. Name the exact fix rather than parsing CSV here.
    const csvPath = join(repoRoot, 'ml', 'artifacts', 'predictions.csv')
    return {
      rows,
      info: {
        available: false,
        hint: existsSync(csvPath)
          ? 'Found predictions.csv. Re-run with --output ml/artifacts/predictions.json'
          : 'Run: debob predict-risk --output ml/artifacts/predictions.json',
      },
    }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(readFileSync(jsonPath, 'utf-8'))
  } catch {
    return { rows, info: { available: false, hint: 'predictions.json could not be parsed' } }
  }
  if (!Array.isArray(parsed)) {
    return { rows, info: { available: false, hint: 'predictions.json is not an array of rows' } }
  }

  let modelName: string | undefined
  for (const entry of parsed) {
    if (typeof entry !== 'object' || entry === null) continue
    const row = entry as Record<string, unknown>
    const filePath = typeof row['file_path'] === 'string' ? row['file_path'] : undefined
    if (!filePath) continue
    const probability = Number(row['risk_probability'])
    if (!Number.isFinite(probability)) continue
    if (typeof row['model'] === 'string') modelName = row['model']

    // `risk_reasons` is a JSON-encoded *string*, not a nested object — predict_risk.py
    // calls json.dumps on it before building the DataFrame, so it survives double-encoded.
    let reasons: RiskReason[] = []
    if (typeof row['risk_reasons'] === 'string') {
      try {
        const decoded: unknown = JSON.parse(row['risk_reasons'])
        if (Array.isArray(decoded)) reasons = decoded as RiskReason[]
      } catch {
        reasons = []
      }
    }

    rows.set(filePath, {
      riskProbability: probability,
      predictedRisky: Number(row['predicted_risky']) === 1,
      riskReasons: reasons,
    })
  }

  // The sidecar carries the scored-candidate count, which the truncated row list cannot.
  let candidateCount: number | undefined
  let stale = false
  try {
    const sidecar = JSON.parse(readFileSync(jsonPath + '.meta.json', 'utf-8')) as Record<string, unknown>
    const candidates = Number(sidecar['candidate_count'])
    if (Number.isFinite(candidates)) candidateCount = candidates
    if (typeof sidecar['model_name'] === 'string') modelName = sidecar['model_name']
    const heads = sidecar['db_head_commits']
    if (Array.isArray(heads) && manifest?.headCommit) {
      stale = heads.length > 0 && !heads.includes(manifest.headCommit)
    }
  } catch {
    // A missing or unreadable sidecar only costs coverage metadata, not the overlay.
  }

  if (rows.size === 0) {
    return { rows, info: { available: false, hint: 'predictions.json contained no usable rows' } }
  }
  return {
    rows,
    info: { available: true, rowsShown: rows.size, candidateCount, modelName, stale },
  }
}

function serialiseNode(
  node: Node,
  enrichment?: { value: string; modelId: string },
  prediction?: PredictionRow,
): VisualiserNode {
  return {
    id: node.id,
    type: node.type,
    name: node.name,
    filePath: node.filePath,
    layer: node.layer,
    startLine: node.startLine,
    endLine: node.endLine,
    confidence: node.confidence,
    dataSource: node.dataSource,
    metadata: node.metadata,
    responsibility: enrichment?.value ?? node.responsibility,
    responsibilityModel: enrichment?.modelId,
    // Left undefined when the file was never scored. The UI must distinguish that from
    // `predictedRisky: false` — only file nodes are ever candidates, so symbol nodes have
    // no prediction rather than a safe one.
    riskProbability: prediction?.riskProbability,
    predictedRisky: prediction?.predictedRisky,
    riskReasons: prediction?.riskReasons,
  }
}

function serialiseEdge(edge: Edge): VisualiserEdge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: edge.type,
    confidence: edge.confidence,
    dataSource: edge.dataSource,
  }
}

function isAddressInUse(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EADDRINUSE'
}

function listen(port: number, payload: VisualiserPayload): Promise<Server> {
  const server = createServer(requestHandler(payload))
  return new Promise((resolve, reject) => {
    const onError = (error: Error) => {
      server.off('listening', onListening)
      reject(error)
    }
    const onListening = () => {
      server.off('error', onError)
      resolve(server)
    }

    server.once('error', onError)
    server.once('listening', onListening)
    server.listen(port, '127.0.0.1')
  })
}

function requestHandler(payload: VisualiserPayload) {
  const graphJson = JSON.stringify(payload)

  return (request: IncomingMessage, response: ServerResponse): void => {
    if (request.method !== 'GET') {
      response.writeHead(405, { Allow: 'GET', 'Content-Type': 'text/plain; charset=utf-8' })
      response.end('Method Not Allowed')
      return
    }

    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname
    if (pathname === '/api/graph') {
      response.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' })
      response.end(graphJson)
      return
    }

    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    response.end(VISUALISER_HTML)
  }
}

// ─── Inline Visualiser ─────────────────────────────────────────────────────

const VISUALISER_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeBob Graph Visualiser</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
  <!--
    fcose is a compound-node-aware layout, which plain 'cose' is not: with region grouping
    on, cose spreads the graph across a ~28000px-wide canvas (zoom 0.034) and strands nodes
    in a line along the top. fcose ships as a webpack UMD bundle that externalises its
    dependencies rather than inlining them, so layout-base and cose-base MUST be loaded
    first and in this order (layoutBase -> coseBase -> cytoscapeFcose). Loading the fcose
    (or cose-bilkent) bundle alone is what produced the earlier, misdiagnosed
    "Cannot read properties of undefined (reading 'layoutBase')" - the package is fine, the
    script tag was just missing its dependencies.
  -->
  <script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
  <style>
    :root { background: #0d141c; color: #e8edf4; color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 960px; overflow: hidden; }
    #app { display: grid; grid-template-columns: 250px minmax(420px, 1fr) 310px; grid-template-rows: 68px calc(100vh - 68px); height: 100vh; }
    #topbar { align-items: center; background: #121d29; border-bottom: 1px solid #253548; display: flex; gap: 18px; grid-column: 1 / -1; padding: 0 22px; }
    .brand { color: white; font-size: 20px; font-weight: 750; white-space: nowrap; }
    .repo-name { color: #8bbcf4; }
    #stats { color: #aebccc; display: flex; font-size: 13px; gap: 14px; white-space: nowrap; }
    #search { background: #0b121a; border: 1px solid #34485d; border-radius: 7px; color: #e8edf4; margin-left: auto; max-width: 360px; outline: none; padding: 9px 12px; width: 32vw; }
    #search:focus { border-color: #4a90d9; box-shadow: 0 0 0 3px rgba(74,144,217,.18); }
    #group-by { background: #0b121a; border: 1px solid #34485d; border-radius: 7px; color: #e8edf4; outline: none; padding: 8px 10px; }
    #group-by:focus { border-color: #4a90d9; box-shadow: 0 0 0 3px rgba(74,144,217,.18); }
    aside { background: #101923; overflow-y: auto; padding: 18px; }
    #filters { border-right: 1px solid #253548; } #inspector { border-left: 1px solid #253548; }
    .panel-title { color: white; font-size: 14px; margin: 0 0 14px; }
    .filter-section { border-top: 1px solid #253548; margin-top: 15px; padding-top: 15px; }
    .filter-section h3 { color: #8fa3b9; font-size: 11px; letter-spacing: .08em; margin: 0 0 9px; text-transform: uppercase; }
    .filter-option { align-items: center; color: #c9d4e0; cursor: pointer; display: flex; font-size: 13px; gap: 8px; margin: 7px 0; }
    .filter-option input { accent-color: #4a90d9; margin: 0; }
    #graph-area { min-width: 0; position: relative; } #cy { height: 100%; width: 100%; }
    #loading { align-items: center; background: #0d141c; color: #b9c6d5; display: flex; inset: 0; justify-content: center; position: absolute; z-index: 2; }
    #loading.error { color: #ff8a8a; padding: 32px; text-align: center; }
    #legend { background: rgba(16,25,35,.94); border: 1px solid #31445a; border-radius: 8px; bottom: 16px; max-width: 330px; padding: 10px 12px; position: absolute; right: 16px; z-index: 1; }
    .legend-title { color: white; font-size: 11px; font-weight: 700; margin: 0 0 6px; text-transform: uppercase; }
    .legend-items { display: flex; flex-wrap: wrap; gap: 6px 11px; }
    .legend-item { align-items: center; color: #c4cfda; display: flex; font-size: 11px; gap: 5px; }
    .legend-swatch { border-radius: 50%; height: 9px; width: 9px; } .legend-line { border-radius: 2px; height: 3px; width: 13px; }
    .marker-enriched { background: #607083; box-shadow: 0 0 0 3px rgba(34,211,238,.45); margin: 0 2px; }
    .marker-hot { background: #607083; box-shadow: 0 0 0 2px #FF0000; margin: 0 1px; }
    .marker-risk { background: #607083; box-shadow: 0 0 0 1px #101923, 0 0 0 3px #FF8A00; margin: 0 2px; }
    .legend-section + .legend-section { margin-top: 9px; }
    .filter-option.disabled { cursor: not-allowed; opacity: .5; }
    .filter-note { color: #8394a7; font-size: 11px; line-height: 1.45; margin: 5px 0 0 22px; overflow-wrap: anywhere; }
    .filter-note.stale { color: #FFB347; }
    .risk-reasons { list-style: none; margin: 7px 0 0; padding: 0; }
    .risk-reasons li { border-bottom: 1px solid #223244; color: #dbe4ee; font-size: 11px; line-height: 1.45; padding: 6px 0; }
    .risk-reasons .reason-value { color: #8fa3b9; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .risk-caveat { color: #7f93a8; font-size: 10px; letter-spacing: .05em; margin: 6px 0 0; text-transform: uppercase; }
    #inspector-empty { color: #8394a7; font-size: 13px; line-height: 1.5; }
    #inspector-content { display: none; } #inspector-content.visible { display: block; }
    .node-heading { color: white; font-size: 15px; margin: 0 0 15px; overflow-wrap: anywhere; }
    .node-summary { background: rgba(74,144,217,.07); border-left: 2px solid #4A90D9; border-radius: 0 5px 5px 0; color: #dbe4ee; font-size: 13px; line-height: 1.55; margin: 0 0 4px; padding: 10px 12px; }
    .summary-attribution { color: #7f93a8; font-size: 10px; letter-spacing: .06em; margin: 0 0 14px; text-transform: uppercase; }
    .detail-row { border-bottom: 1px solid #223244; padding: 8px 0; }
    .detail-label { color: #8496aa; display: block; font-size: 10px; letter-spacing: .07em; margin-bottom: 3px; text-transform: uppercase; }
    .detail-value, .edge-list li { color: #dbe4ee; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .connected-title { color: white; font-size: 13px; margin: 18px 0 0; }
    .edge-list { list-style: none; margin: 7px 0 0; padding: 0; }
    .edge-list li { border-bottom: 1px solid #223244; font-size: 11px; line-height: 1.45; padding: 7px 0; }
    #edge-tooltip { background: rgba(16,25,35,.96); border: 1px solid #31445a; border-radius: 6px; color: #dbe4ee; display: none; font-size: 12px; max-width: 320px; overflow-wrap: anywhere; padding: 8px 10px; pointer-events: none; position: absolute; z-index: 3; }
    #edge-tooltip .tip-type { color: #8fa3b9; font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }
  </style>
</head>
<body>
  <div id="app">
    <header id="topbar">
      <div class="brand">DeBob <span class="repo-name" id="repo-name">Graph</span></div>
      <div id="stats"><span id="node-count">0 nodes</span><span id="edge-count">0 edges</span><span id="commit-count">0 commits</span></div>
      <select id="group-by" aria-label="Group nodes into regions">
        <option value="none">No grouping</option>
        <option value="folder" selected>Group by folder</option>
        <option value="layer">Group by layer</option>
      </select>
      <input id="search" type="search" aria-label="Search graph nodes" placeholder="Search node names">
    </header>
    <aside id="filters">
      <h2 class="panel-title">Filters</h2>
      <div id="node-type-filters"></div><div id="edge-type-filters"></div><div id="layer-filters"></div>
      <div class="filter-section"><label class="filter-option"><input id="hot-only" type="checkbox"> Hot files only</label></div>
      <div class="filter-section">
        <h3>Machine learning</h3>
        <label class="filter-option" id="risk-overlay-label"><input id="risk-overlay" type="checkbox"> Predicted risky files only</label>
        <p class="filter-note" id="risk-note"></p>
      </div>
    </aside>
    <main id="graph-area">
      <div id="loading">Loading graph…</div><div id="cy"></div>
      <div id="edge-tooltip"></div>
      <section id="legend">
        <div class="legend-section"><div class="legend-title">Node types</div><div class="legend-items" id="node-legend"></div></div>
        <div class="legend-section"><div class="legend-title">Markers</div><div class="legend-items">
          <div class="legend-item"><span class="legend-swatch marker-enriched"></span><span>described by watsonx</span></div>
          <div class="legend-item"><span class="legend-swatch marker-hot"></span><span>hot (high churn)</span></div>
          <div class="legend-item" id="risk-legend-item" style="display:none"><span class="legend-swatch marker-risk"></span><span id="risk-legend-text">predicted risky</span></div>
        </div></div>
        <div class="legend-section"><div class="legend-title">Edge types</div><div class="legend-items" id="edge-legend"></div></div>
      </section>
    </main>
    <aside id="inspector"><h2 class="panel-title">Node Inspector</h2><div id="inspector-empty">Select a node to inspect its graph data and connections.</div><div id="inspector-content"></div></aside>
  </div>
  <script>
    (function () {
      var NODE_TYPES = ['file', 'class', 'function', 'interface', 'variable', 'package', 'route'];
      var EDGE_TYPES = ['imports', 'exports', 'declares', 'extends', 'implements', 'calls', 'depends_on', 'instantiates', 'exposes', 'handles', 'tests', 'reads_from', 'writes_to', 'communicates_with', 'configured_by', 'related_to'];
      var LAYERS = ['presentation', 'business', 'data', 'config', 'test', 'infra', 'unclassified'];
      var NODE_COLORS = { file: '#4A90D9', class: '#7B61FF', function: '#50C878', interface: '#FFB347', variable: '#87CEEB', package: '#FF6B6B', route: '#FFD700' };
      var EDGE_COLORS = {
        imports: '#9AA5B1', exports: '#4A90D9', declares: '#3D5068', extends: '#7B61FF', implements: '#FFB347',
        calls: '#50C878', depends_on: '#FF6B6B', instantiates: '#E879F9', exposes: '#22D3EE',
        handles: '#F97316', tests: '#A3E635', reads_from: '#38BDF8', writes_to: '#F43F5E',
        communicates_with: '#C084FC', configured_by: '#FACC15', related_to: '#94A3B8'
      };
      var DEFAULT_EDGE_COLOR = '#94A3B8';
      var state = { nodeTypes: new Set(NODE_TYPES), edgeTypes: new Set(EDGE_TYPES), layers: new Set(LAYERS), hotOnly: false, riskyOnly: false, groupBy: 'folder' };
      var graphData = null;
      var cy = null;
      var search = document.getElementById('search');
      var groupBySelect = document.getElementById('group-by');
      var loading = document.getElementById('loading');
      var inspector = document.getElementById('inspector-content');
      var inspectorEmpty = document.getElementById('inspector-empty');
      var edgeTooltip = document.getElementById('edge-tooltip');
      var cyContainer = document.getElementById('cy');

      // fcose self-registers when a global cytoscape already exists (it does - cytoscape
      // loads first), but register explicitly too so a load-order change can't silently
      // downgrade us to the plain-cose fallback again.
      if (window.cytoscape && window.cytoscapeFcose) {
        try { window.cytoscape.use(window.cytoscapeFcose); } catch (registerError) { /* already registered */ }
      }

      // ── Region grouping (compound nodes) ─────────────────────────────────
      var CONTAINER_DIRS = ['packages', 'apps', 'libs', 'services', 'modules'];

      function folderRegionFor(node) {
        if (node.type === 'package') return { id: 'region:external', label: 'External Packages' };
        var parts = String(node.filePath || node.id || '').split('/').filter(Boolean);
        if (parts.length <= 1) return { id: 'region:root', label: '(root)' };
        if (CONTAINER_DIRS.indexOf(parts[0]) !== -1 && parts.length >= 2) {
          var label = parts[0] + '/' + parts[1];
          return { id: 'region:' + label, label: label };
        }
        return { id: 'region:' + parts[0], label: parts[0] };
      }

      function layerRegionFor(node) {
        if (node.type === 'package') return { id: 'region:external', label: 'External Packages' };
        var layer = node.layer || 'unclassified';
        return { id: 'region:layer:' + layer, label: layer };
      }

      function regionFor(node) {
        if (state.groupBy === 'layer') return layerRegionFor(node);
        if (state.groupBy === 'folder') return folderRegionFor(node);
        return null;
      }

      function filterGroup(containerId, title, values, selected) {
        var container = document.getElementById(containerId);
        var section = document.createElement('section');
        section.className = 'filter-section';
        var heading = document.createElement('h3');
        heading.textContent = title;
        section.appendChild(heading);
        values.forEach(function (value) {
          var label = document.createElement('label');
          label.className = 'filter-option';
          var checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.checked = true;
          checkbox.addEventListener('change', function () {
            if (checkbox.checked) selected.add(value);
            else selected.delete(value);
            applyFilters();
          });
          label.appendChild(checkbox);
          label.appendChild(document.createTextNode(value));
          section.appendChild(label);
        });
        container.appendChild(section);
      }

      function legend(containerId, values, colors, line) {
        var container = document.getElementById(containerId);
        values.forEach(function (value) {
          var item = document.createElement('div');
          item.className = 'legend-item';
          var swatch = document.createElement('span');
          swatch.className = line ? 'legend-line' : 'legend-swatch';
          swatch.style.backgroundColor = colors[value] || DEFAULT_EDGE_COLOR;
          var label = document.createElement('span');
          label.textContent = value;
          item.appendChild(swatch);
          item.appendChild(label);
          container.appendChild(item);
        });
      }

      filterGroup('node-type-filters', 'Node type', NODE_TYPES, state.nodeTypes);
      filterGroup('edge-type-filters', 'Edge type', EDGE_TYPES, state.edgeTypes);
      filterGroup('layer-filters', 'Layer', LAYERS, state.layers);
      legend('node-legend', NODE_TYPES, NODE_COLORS, false);
      legend('edge-legend', EDGE_TYPES, EDGE_COLORS, true);
      document.getElementById('hot-only').addEventListener('change', function (event) {
        state.hotOnly = event.target.checked;
        applyFilters();
      });
      document.getElementById('risk-overlay').addEventListener('change', function (event) {
        state.riskyOnly = event.target.checked;
        document.getElementById('risk-legend-item').style.display = event.target.checked ? '' : 'none';
        // Filter + ring, but never a rebuild: applyFilters only shows/hides elements, so the
        // layout positions the user is looking at survive the toggle.
        applyFilters();
        applyRiskOverlay();
        // The visible set collapses from hundreds of nodes to a handful, and hidden nodes
        // keep their layout positions - without a refit the survivors are stranded off
        // screen at the old zoom. Refit on the way back out too.
        var shown = cy.nodes(':visible');
        if (shown.length) cy.animate({ fit: { eles: shown, padding: 60 }, duration: 260 });
      });
      search.addEventListener('input', applyFilters);
      groupBySelect.addEventListener('change', function (event) {
        state.groupBy = event.target.value;
        // Stop any in-flight layout before destroying: an async layout that ticks after
        // its cytoscape instance is gone throws "Cannot read properties of null
        // (reading 'isHeadless')" straight onto the page.
        if (graphData) {
          if (cy) { cy.elements().stop(true); cy.destroy(); cy = null; }
          setupGraph(graphData);
        }
      });

      function setupGraph(data) {
        graphData = data;
        var manifest = data.manifest;
        var parts = manifest && manifest.repoPath ? manifest.repoPath.split(/[\\\\/]/).filter(Boolean) : [];
        document.getElementById('repo-name').textContent = parts[parts.length - 1] || 'Graph';
        document.getElementById('node-count').textContent = (manifest ? manifest.nodeCount : data.nodes.length) + ' nodes';
        document.getElementById('edge-count').textContent = (manifest ? manifest.edgeCount : data.edges.length) + ' edges';
        document.getElementById('commit-count').textContent = (manifest ? manifest.commitCount : 0) + ' commits';
        setupRiskControls(data.predictionInfo);

        var churns = data.nodes.filter(function (node) { return node.type === 'file'; }).map(function (node) { return Number(node.metadata && node.metadata.churnScore) || 0; });
        var maxChurn = Math.max.apply(Math, [0].concat(churns));
        var elements = [];

        // Regions with only one member produce a trivial single-node compound parent.
        // fcose lays out many tiny compound parents mixed with a few large ones very
        // badly (they collapse into a stray line while the real cluster gets squeezed
        // into a corner) - so only box regions with more than one member; a singleton's
        // sole node just stays an ordinary top-level node.
        var regionCounts = {};
        var regionLabels = {};
        data.nodes.forEach(function (node) {
          var region = regionFor(node);
          if (!region) return;
          regionCounts[region.id] = (regionCounts[region.id] || 0) + 1;
          regionLabels[region.id] = region.label;
        });

        data.nodes.forEach(function (node) {
          var churn = Number(node.metadata && node.metadata.churnScore) || 0;
          var radius = node.type === 'file' ? 20 + (maxChurn > 0 ? Math.max(0, churn) / maxChurn * 40 : 0) : 18;
          var region = regionFor(node);
          var nodeData = Object.assign({}, node, { label: node.name, diameter: radius * 2 });
          if (region && regionCounts[region.id] > 1) nodeData.parent = region.id;
          elements.push({
            group: 'nodes',
            data: nodeData,
            classes: 'node-' + node.type
              + (node.metadata && node.metadata.hot === true ? ' hot' : '')
              + (node.responsibility ? ' enriched' : '')
          });
        });
        // Region (compound parent) nodes must exist before their children reference them.
        Object.keys(regionLabels).forEach(function (regionId) {
          if (regionCounts[regionId] > 1) {
            elements.unshift({ group: 'nodes', data: { id: regionId, label: regionLabels[regionId] }, classes: 'region' });
          }
        });
        data.edges.forEach(function (edge) {
          elements.push({
            group: 'edges',
            data: Object.assign({}, edge, { edgeColor: EDGE_COLORS[edge.type] || DEFAULT_EDGE_COLOR }),
            classes: 'edge-' + edge.type
          });
        });

        var styles = [
          { selector: 'node', style: { 'background-color': '#607083', 'border-color': '#101923', 'border-width': 1, color: '#e8edf4', label: 'data(label)', 'font-size': 10, 'min-zoomed-font-size': 8, 'text-outline-color': '#0d141c', 'text-outline-width': 2, 'text-valign': 'bottom', 'text-margin-y': 5 } },
          // Scoped to [diameter] so region compound nodes - which have no diameter and size
          // themselves from their children - don't trigger a mapping warning per element.
          { selector: 'node[diameter]', style: { height: 'data(diameter)', width: 'data(diameter)' } },
          { selector: 'edge', style: { 'curve-style': 'bezier', opacity: .72, 'target-arrow-shape': 'triangle', width: 1.6 } },
          // Scoped to [edgeColor] so the structural anchors, which carry no colour, don't
          // each emit a pair of mapping warnings.
          { selector: 'edge[edgeColor]', style: { 'line-color': 'data(edgeColor)', 'target-arrow-color': 'data(edgeColor)' } },
          { selector: 'edge.edge-hover', style: { opacity: 1, width: 2.8, 'z-index': 998 } },
          // declares edges are the graph's connective tissue and vastly outnumber the rest.
          // Drawn faintly and without an arrowhead so they shape the layout and stay
          // inspectable, without turning the canvas into a hairball.
          { selector: 'edge.edge-declares', style: { opacity: .22, width: 1, 'target-arrow-shape': 'none' } },
          { selector: 'edge.edge-declares.edge-hover', style: { opacity: .9, width: 2.4 } },
          // A soft halo rather than a border, so "watsonx described this" composes with the
          // red hot-file border instead of fighting it for the same visual channel.
          { selector: 'node.enriched', style: { 'underlay-color': '#22D3EE', 'underlay-opacity': .28, 'underlay-padding': 6 } },
          { selector: 'node.hot', style: { 'border-color': '#FF0000', 'border-width': 3 } },
          // An offset outline, not a border: 'hot' already owns the border channel, and a
          // file can be both. Orange rather than a second red so the two stay tellable
          // apart - which is the interesting case, since the operational model excludes
          // churn, so "risky but not hot" is exactly what the ML adds over the heuristic.
          { selector: 'node.risk-high', style: { 'outline-color': '#FF8A00', 'outline-width': 4, 'outline-offset': 2, 'outline-opacity': 1 } },
          { selector: 'node.search-dim', style: { opacity: .13 } },
          { selector: 'node.search-match', style: { 'border-color': '#ffffff', 'border-width': 4, 'z-index': 999 } },
          { selector: 'node.region', style: {
            'background-color': '#4A90D9', 'background-opacity': .06,
            'border-color': '#31445a', 'border-style': 'dashed', 'border-width': 1,
            color: '#c3d2e2', 'font-size': 26, 'font-weight': 600, label: 'data(label)',
            // Grouped mode fits at ~0.4-0.55 zoom, where the base 'min-zoomed-font-size: 8'
            // would hide these labels entirely - which is the one thing grouping is for.
            'min-zoomed-font-size': 0, padding: 22,
            shape: 'round-rectangle', 'text-halign': 'center', 'text-margin-y': -8,
            'text-outline-color': '#0d141c', 'text-outline-width': 3, 'text-valign': 'top'
          } }
        ];
        NODE_TYPES.forEach(function (type) {
          styles.push({ selector: 'node.node-' + type, style: { 'background-color': NODE_COLORS[type] } });
        });

        var grouped = state.groupBy !== 'none';
        var fcoseLayout = {
          name: 'fcose',
          animate: false,
          quality: 'proof',
          randomize: true,
          idealEdgeLength: 85,
          // Compound parents need markedly less repulsion than free nodes, otherwise each
          // region inflates until the regions themselves no longer fit beside each other.
          nodeRepulsion: grouped ? 6000 : 9000,
          nodeSeparation: 90,
          gravity: grouped ? 0.9 : 0.35,
          gravityRange: 3,
          gravityCompound: 2.2,
          gravityRangeCompound: 1.2,
          numIter: 3500,
          tile: true,
          tilingPaddingVertical: 14,
          tilingPaddingHorizontal: 14,
          packComponents: true,
          padding: 40
        };
        var coseLayout = {
          // Fallback only if the layout CDN scripts failed to load (e.g. offline demo).
          // Plain cose is not compound-aware, so grouped mode degrades noticeably here.
          name: 'cose',
          animate: false,
          idealEdgeLength: 90,
          nodeRepulsion: 7000,
          gravity: 1,
          numIter: 1500,
          padding: 50,
          tile: true
        };

        var layoutName = 'fcose';
        try {
          cy = window.cytoscape({ container: cyContainer, elements: elements, style: styles, layout: fcoseLayout, wheelSensitivity: .2 });
        } catch (fcoseError) {
          console.warn('fcose layout unavailable, falling back to cose:', fcoseError);
          if (cy) cy.destroy();
          layoutName = 'cose';
          cy = window.cytoscape({ container: cyContainer, elements: elements, style: styles, layout: coseLayout, wheelSensitivity: .2 });
        }
        window.__debobLayoutName = layoutName;
        // Regardless of layout, frame the result to the viewport so the graph is never
        // left at a stray zoom level the user has to hunt around in.
        cy.one('layoutstop', function () { cy.fit(undefined, 40); });
        cy.on('tap', 'node', function (event) {
          if (event.target.isParent()) return;
          showInspector(event.target.data('id'));
        });
        // 'edge[type]' matches only real graph edges: the layout-only structural anchors
        // carry no type field. (cytoscape has no :not() pseudo-class - selecting on the
        // absence of a data field is the supported idiom.)
        cy.on('tap', 'edge[type]', function (event) { showEdgeInspector(event.target.data()); });
        cy.on('tap', function (event) { if (event.target === cy) clearInspector(); });
        cy.on('mouseover', 'edge[type]', function (event) {
          event.target.addClass('edge-hover');
          showEdgeTooltip(event);
        });
        cy.on('mousemove', 'edge[type]', function (event) { positionEdgeTooltip(event); });
        cy.on('mouseout', 'edge[type]', function (event) {
          event.target.removeClass('edge-hover');
          hideEdgeTooltip();
        });
        applyFilters();
        // Re-assert the overlay: changing group-by destroys and rebuilds the instance,
        // which would otherwise drop the risk classes while the checkbox stays ticked.
        applyRiskOverlay();
        loading.style.display = 'none';
      }

      /**
       * Enables or disables the ML toggle and states the overlay's coverage.
       *
       * ml/artifacts/ is gitignored, so a fresh clone has the graph but no predictions -
       * the disabled state is the normal path, not an error, and it names the command that
       * would populate it.
       */
      function setupRiskControls(info) {
        var checkbox = document.getElementById('risk-overlay');
        var label = document.getElementById('risk-overlay-label');
        var note = document.getElementById('risk-note');
        if (!info || !info.available) {
          checkbox.checked = false;
          checkbox.disabled = true;
          state.riskyOnly = false;
          label.classList.add('disabled');
          note.textContent = (info && info.hint) || 'No predictions found.';
          return;
        }
        checkbox.disabled = false;
        label.classList.remove('disabled');

        // --top truncates the rows predict_risk.py *writes*, not just what it prints, so
        // the file usually holds a handful of a much larger scored set. Say so plainly
        // rather than letting the overlay imply it covers everything.
        var coverage = info.candidateCount && info.candidateCount > info.rowsShown
          ? 'top ' + info.rowsShown + ' of ' + info.candidateCount + ' scored'
          : info.rowsShown + ' scored';
        document.getElementById('risk-legend-text').textContent = 'predicted risky (' + coverage + ')';
        var text = coverage + (info.modelName ? ' · ' + info.modelName : '');
        if (info.stale) {
          text += ' · STALE: scored against a different graph — re-run debob predict-risk';
          note.classList.add('stale');
        } else {
          note.classList.remove('stale');
        }
        note.textContent = text;
      }

      /**
       * Rings the files the model predicted risky.
       *
       * Only file nodes are ever scored, so a node with no riskProbability has *no data* -
       * it is never styled as "predicted safe".
       */
      function applyRiskOverlay() {
        if (!cy) return;
        cy.nodes().forEach(function (element) {
          if (element.hasClass('region')) return;
          var node = element.data();
          var risky = state.riskyOnly && node.predictedRisky === true;
          if (risky) element.addClass('risk-high'); else element.removeClass('risk-high');
        });
      }

      function applyFilters() {
        if (!cy || !graphData) return;
        var visibleNodes = new Set();
        var regions = [];
        cy.nodes().forEach(function (element) {
          // Regions are decided after their children, below - an empty box is worse than
          // no box, and 'risky files only' empties most of them.
          if (element.hasClass('region')) { regions.push(element); return; }
          var node = element.data();
          var layer = node.layer || 'unclassified';
          var hot = node.metadata && node.metadata.hot === true;
          // Only scored file nodes can satisfy riskyOnly. Symbols and unscored files have
          // no prediction, so they drop out - "no data" is not "predicted safe".
          var risky = node.predictedRisky === true;
          var visible = state.nodeTypes.has(node.type) && state.layers.has(layer)
            && (!state.hotOnly || hot) && (!state.riskyOnly || risky);
          if (visible) {
            visibleNodes.add(node.id);
            element.show();
          } else element.hide();
        });
        regions.forEach(function (element) {
          var hasVisibleChild = element.children().some(function (child) { return child.visible(); });
          if (hasVisibleChild) element.show(); else element.hide();
        });
        cy.edges().forEach(function (element) {
          var edge = element.data();
          if (state.edgeTypes.has(edge.type) && visibleNodes.has(edge.source) && visibleNodes.has(edge.target)) element.show();
          else element.hide();
        });
        var query = search.value.trim().toLowerCase();
        cy.nodes().removeClass('search-dim search-match');
        if (query) cy.nodes(':visible').forEach(function (element) {
          if (element.hasClass('region')) return;
          var name = String(element.data('name') || '').toLowerCase();
          element.addClass(name.indexOf(query) >= 0 ? 'search-match' : 'search-dim');
        });
      }

      function detail(label, value) {
        var row = document.createElement('div');
        row.className = 'detail-row';
        var key = document.createElement('span');
        key.className = 'detail-label';
        key.textContent = label;
        var text = document.createElement('span');
        text.className = 'detail-value';
        text.textContent = value === undefined || value === null || value === '' ? '—' : String(value);
        row.appendChild(key);
        row.appendChild(text);
        inspector.appendChild(row);
      }

      function metadata(node, key) {
        return node.metadata && Object.prototype.hasOwnProperty.call(node.metadata, key) ? node.metadata[key] : '—';
      }

      function lineRange(start, end) {
        if (start === undefined && end === undefined) return '—';
        if (start === undefined) return String(end);
        if (end === undefined || end === start) return String(start);
        return start + '–' + end;
      }

      function showInspector(nodeId) {
        var node = graphData.nodes.find(function (item) { return item.id === nodeId; });
        if (!node) return;
        inspector.replaceChildren();
        inspectorEmpty.style.display = 'none';
        inspector.classList.add('visible');
        var title = document.createElement('h3');
        title.className = 'node-heading';
        title.textContent = node.name;
        inspector.appendChild(title);
        // The LLM's summary is the most valuable thing on this panel, so it leads - as prose,
        // visually distinct from the deterministic monospace facts below, and attributed so
        // it's never mistaken for something static analysis derived. Absent enrichment shows
        // nothing at all rather than an empty placeholder row.
        if (node.responsibility) {
          var summary = document.createElement('p');
          summary.className = 'node-summary';
          summary.textContent = node.responsibility;
          inspector.appendChild(summary);
          var attribution = document.createElement('div');
          attribution.className = 'summary-attribution';
          attribution.textContent = 'watsonx' + (node.responsibilityModel ? ' · ' + node.responsibilityModel : '');
          inspector.appendChild(attribution);
        }
        detail('ID', node.id); detail('Name', node.name); detail('Type', node.type);
        detail('File path', node.filePath); detail('Layer', node.layer || 'unclassified');
        detail('Lines', lineRange(node.startLine, node.endLine)); detail('Confidence', node.confidence);
        detail('Data source', node.dataSource); detail('Churn score', metadata(node, 'churnScore'));
        detail('Author count', metadata(node, 'authorCount')); detail('Last modified', metadata(node, 'lastModifiedAt'));
        detail('Hot', metadata(node, 'hot'));
        // Risk fields only render for scored files. A node with no prediction shows no risk
        // rows at all rather than a "no"/"0.00", which would read as "the model cleared it".
        if (typeof node.riskProbability === 'number') {
          detail('Risk probability', node.riskProbability.toFixed(3));
          detail('Predicted risky', node.predictedRisky ? 'yes' : 'no');
          if (node.riskReasons && node.riskReasons.length) {
            var reasonsHeading = document.createElement('h4');
            reasonsHeading.className = 'connected-title';
            reasonsHeading.textContent = 'Influential signals';
            inspector.appendChild(reasonsHeading);
            var reasonList = document.createElement('ul');
            reasonList.className = 'risk-reasons';
            node.riskReasons.forEach(function (reason) {
              var item = document.createElement('li');
              item.textContent = reason.label;
              var value = document.createElement('span');
              value.className = 'reason-value';
              value.textContent = ' · ' + reason.value;
              item.appendChild(value);
              reasonList.appendChild(item);
            });
            inspector.appendChild(reasonList);
            // Global feature importance plus a threshold check - not per-file attribution.
            // The model card makes the same distinction; the UI must not overstate it.
            var caveat = document.createElement('p');
            caveat.className = 'risk-caveat';
            caveat.textContent = 'Model signals, not causal explanations';
            inspector.appendChild(caveat);
          }
        }
        var connected = graphData.edges.filter(function (edge) { return edge.source === node.id || edge.target === node.id; });
        var heading = document.createElement('h4');
        heading.className = 'connected-title';
        heading.textContent = 'Connected edges (' + connected.length + ')';
        inspector.appendChild(heading);
        if (!connected.length) { detail('Connections', 'None'); return; }
        var list = document.createElement('ul');
        list.className = 'edge-list';
        connected.forEach(function (edge) {
          var item = document.createElement('li');
          item.textContent = edge.source + ' → ' + edge.type + ' → ' + edge.target;
          list.appendChild(item);
        });
        inspector.appendChild(list);
      }

      function clearInspector() {
        inspector.replaceChildren();
        inspector.classList.remove('visible');
        inspectorEmpty.style.display = 'block';
      }

      function showEdgeInspector(edge) {
        inspector.replaceChildren();
        inspectorEmpty.style.display = 'none';
        inspector.classList.add('visible');
        var title = document.createElement('h3');
        title.className = 'node-heading';
        title.textContent = edge.type + ' edge';
        inspector.appendChild(title);
        detail('Source', edge.source);
        detail('Target', edge.target);
        detail('Type', edge.type);
        detail('Confidence', edge.confidence);
        detail('Data source', edge.dataSource);
      }

      function showEdgeTooltip(event) {
        var edge = event.target.data();
        edgeTooltip.replaceChildren();
        var typeLine = document.createElement('div');
        typeLine.className = 'tip-type';
        typeLine.textContent = edge.type;
        var pathLine = document.createElement('div');
        pathLine.textContent = edge.source + ' → ' + edge.target;
        edgeTooltip.appendChild(typeLine);
        edgeTooltip.appendChild(pathLine);
        edgeTooltip.style.display = 'block';
        positionEdgeTooltip(event);
      }

      function positionEdgeTooltip(event) {
        var original = event.originalEvent;
        if (!original) return;
        var rect = cyContainer.getBoundingClientRect();
        edgeTooltip.style.left = (original.clientX - rect.left + 14) + 'px';
        edgeTooltip.style.top = (original.clientY - rect.top + 14) + 'px';
      }

      function hideEdgeTooltip() {
        edgeTooltip.style.display = 'none';
      }

      fetch('/api/graph')
        .then(function (response) {
          if (!response.ok) throw new Error('Graph API returned status ' + response.status);
          return response.json();
        })
        .then(setupGraph)
        .catch(function (error) {
          loading.classList.add('error');
          loading.textContent = 'Unable to load graph: ' + error.message;
        });
    }());
  </script>
</body>
</html>
`
