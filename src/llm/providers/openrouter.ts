import type {
  DiffContext,
  LLMAdapter,
  LLMConfig,
  ModuleContext,
  ModuleDescription,
  QueryContext,
  TokenUsage,
} from '../adapter.js'
import { parseModuleDescription } from './watsonx.js'

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
const DEFAULT_MODEL = 'google/gemma-4-31b-it:free'

type Message = { role: 'system' | 'user'; content: string }

function buildModulePrompt(ctx: ModuleContext): string {
  const lines: string[] = [`File: ${ctx.filePath}`]
  if (ctx.layer) lines.push(`Layer (heuristic): ${ctx.layer}`)
  if (ctx.doc) lines.push('', 'Module documentation:', `  ${ctx.doc}`)

  lines.push(
    '',
    `Imports (${ctx.imports.length}):`,
    ...ctx.imports.map(item => `  - ${item}`),
    '',
    `Declarations (${ctx.declarations.length}):`,
    ...ctx.declarations.map(declaration => {
      const where = declaration.startLine != null ? ` (line ${declaration.startLine})` : ''
      const doc = declaration.doc ? ` — ${declaration.doc}` : ''
      return `  - ${declaration.type} ${declaration.name}${where}${doc}`
    }),
  )

  if (ctx.reExports.length > 0) lines.push('', `Re-exports from (${ctx.reExports.length}):`, ...ctx.reExports.map(item => `  - ${item}`))
  if (ctx.calls && ctx.calls.length > 0) lines.push('', `Calls into (${ctx.calls.length}):`, ...ctx.calls.map(item => `  - ${item}`))
  if (ctx.calledBy && ctx.calledBy.length > 0) lines.push('', `Called by (${ctx.calledBy.length}):`, ...ctx.calledBy.map(item => `  - ${item}`))
  if (ctx.gitStats) {
    lines.push(
      '',
      'Git stats:',
      `  - Churn score   : ${ctx.gitStats.churnScore}`,
      `  - Author count  : ${ctx.gitStats.authorCount}`,
      `  - Last modified : ${ctx.gitStats.lastModifiedAt}`,
    )
  }
  return lines.join('\n')
}

function describeDiff(context: DiffContext): string {
  const nodeLines = [...context.affectedNodes, ...context.neighbourhood.nodes.values()]
    .map(node => `  - ${node.id}${node.layer ? ` [${node.layer}]` : ''} (type: ${node.type})`)
  return [
    context.layersSummary.length > 0
      ? `Layers affected: ${context.layersSummary.join(', ')}`
      : 'Layers affected: unknown',
    '',
    `Directly changed files (${context.affectedNodes.length}):`,
    ...context.affectedNodes.map(node => `  - ${node.id}${node.layer ? ` [${node.layer}]` : ''}`),
    '',
    `Neighbourhood (2-hop, ${context.neighbourhood.nodes.size} nodes):`,
    ...nodeLines,
    '',
    '--- diff (truncated to 200 lines) ---',
    context.diff,
  ].join('\n')
}

export class OpenRouterProvider implements LLMAdapter {
  readonly provider = 'openrouter'
  private readonly apiKey: string
  private readonly model: string
  private usage: TokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0, callCount: 0 }

  constructor(_config?: LLMConfig) {
    const apiKey = process.env['OPENROUTER_API_KEY']
    if (!apiKey) throw new Error('OpenRouterProvider: OPENROUTER_API_KEY is required')
    this.apiKey = apiKey
    this.model = process.env['OPENROUTER_MODEL'] || DEFAULT_MODEL
  }

  async describeModule(context: ModuleContext, preamble?: string): Promise<ModuleDescription> {
    const raw = await this.chat([
      {
        role: 'system',
        content:
          'You are a software architecture assistant. You are given structural facts about one module. ' +
          (preamble ? `\n\nProject context:\n${preamble}\n\n` : '') +
          'Reply with ONLY a JSON object, no prose and no code fence:\n' +
          '{"responsibility": "<1-3 sentences on what this module is for>", "layer": "<one of: presentation, business, data, config, test, infra>"}',
      },
      { role: 'user', content: buildModulePrompt(context) },
    ])
    const parsed = parseModuleDescription(raw)
    if (!parsed) throw new Error('OpenRouterProvider: describeModule response was not parseable JSON')
    return parsed
  }

  summarizeModule(context: ModuleContext): Promise<string> {
    return this.chat([
      { role: 'system', content: 'You are a software architecture assistant. Given module metadata (no source code), write one concise sentence describing the module\'s primary responsibility.' },
      { role: 'user', content: `${buildModulePrompt(context)}\n\nResponsibility:` },
    ])
  }

  async classifyLayer(context: ModuleContext): Promise<string> {
    const raw = await this.chat([
      { role: 'system', content: 'You are a software architecture assistant. Given module metadata (no source code), classify the module into exactly one of these architectural layers: presentation, business, data, config, test, infra. Respond with only the single layer name — no explanation, no punctuation.' },
      { role: 'user', content: `${buildModulePrompt(context)}\n\nLayer:` },
    ])
    return raw.trim().toLowerCase().replace(/[^a-z]/g, '') || 'unclassified'
  }

  explainDiff(context: DiffContext): Promise<string> {
    return this.chat([
      { role: 'system', content: 'You are a software architecture assistant. Given a git diff and the architectural context of affected modules (no raw source), describe which parts of the system are affected, what risks the change introduces, and which neighbouring modules may need review.' },
      { role: 'user', content: describeDiff(context) },
    ])
  }

  answerQuestion(context: QueryContext): Promise<string> {
    if (context.relevantNodes.length === 0) return Promise.resolve("I couldn't find anything in the repository graph relevant to that question.")
    const nodes = context.relevantNodes.map(node => `  - ${node.id} (type: ${node.type})${node.layer ? ` [${node.layer}]` : ''}${node.responsibility ? ` — ${node.responsibility}` : ''}`)
    const edges = context.relevantEdges.map(edge => `  - ${edge.source} --${edge.type}--> ${edge.target}`)
    return this.chat([
      { role: 'system', content: 'You are a software architecture assistant answering questions about a codebase. You are given structured graph metadata, never raw source code. Answer the question grounded only in this data. If the data is insufficient, say so plainly.' },
      { role: 'user', content: [`Question: ${context.question}`, '', `Relevant nodes (${nodes.length}):`, ...nodes, '', `Relevant edges (${edges.length}):`, ...(edges.length > 0 ? edges : ['  (none)'])].join('\n') },
    ])
  }

  getUsage(): TokenUsage | undefined {
    return this.usage.callCount === 0 ? undefined : { ...this.usage }
  }

  private async chat(messages: Message[]): Promise<string> {
    let response: Response
    try {
      response = await fetch(OPENROUTER_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          messages: [{ role: 'user', content: messages.map(message => message.content).join('\n\n') }],
        }),
      })
    } catch (error) {
      throw new Error(`OpenRouter request failed: ${error instanceof Error ? error.message : String(error)}`)
    }

    const body = await response.text()
    if (!response.ok) {
      const detail = body.slice(0, 500) || response.statusText || 'no response body'
      const suffix = response.status === 429 ? ' (rate limit exceeded)' : ''
      throw new Error(`OpenRouter request failed with HTTP ${response.status}${suffix}: ${detail}`)
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(body)
    } catch {
      throw new Error('OpenRouter returned an invalid JSON response')
    }

    const result = parsed as { choices?: Array<{ message?: { content?: unknown } }>; usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } }
    const usage = result.usage
    if (usage) {
      this.usage.promptTokens += usage.prompt_tokens ?? 0
      this.usage.completionTokens += usage.completion_tokens ?? 0
      this.usage.totalTokens += usage.total_tokens ?? 0
      this.usage.callCount += 1
    }
    const content = result.choices?.[0]?.message?.content
    if (typeof content !== 'string' || content.length === 0) throw new Error(`OpenRouter returned an unexpected response shape: ${body.slice(0, 500)}`)
    return content.trim()
  }
}

export { DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL }
