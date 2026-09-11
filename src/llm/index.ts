import type { LLMAdapter } from './adapter.js'
import type { LLMConfig } from './adapter.js'
import { WatsonxProvider } from './providers/watsonx.js'
import { OpenAIProvider } from './providers/openai.js'
import { OpenRouterProvider } from './providers/openrouter.js'
import type { DiffContext, ModuleContext, ModuleDescription, QueryContext, TokenUsage } from './adapter.js'

// ─── LLM Adapter Factory ──────────────────────────────────────────────────────

/** One provider in the fallback chain, plus why it is unavailable when it is. */
interface Candidate {
  /** Display name used in the combined error message. */
  name: string
  /** Constructed adapter, absent when construction failed. */
  adapter?: LLMAdapter
  /** Construction failure, kept so the combined error can explain it. */
  error?: Error
}

function candidate(name: string, construct: () => LLMAdapter): Candidate {
  try {
    return { name, adapter: construct() }
  } catch (error) {
    return { name, error: asError(error) }
  }
}

/**
 * Create an LLMAdapter for the given provider.
 *
 * V1 supported providers:
 *  - `"watsonx"` → watsonx, falling back to OpenAI then OpenRouter
 *
 * The returned adapter tries each configured provider in order and uses the first that
 * answers, so a dead watsonx key degrades to OpenAI or OpenRouter instead of failing.
 * Providers whose credentials are absent are skipped rather than treated as errors.
 *
 * @throws If no provider in the chain could be constructed.
 */
export function createLLMAdapter(provider: string, config: LLMConfig): LLMAdapter {
  switch (provider) {
    case 'watsonx': {
      const candidates = [
        candidate('Watsonx', () => new WatsonxProvider(config)),
        candidate('OpenAI', () => new OpenAIProvider()),
        candidate('OpenRouter', () => new OpenRouterProvider()),
      ]
      if (!candidates.some(entry => entry.adapter)) throw combinedProviderError(candidates)
      return new FallbackLLMAdapter(candidates)
    }
    default:
      throw new Error(
        `createLLMAdapter: unknown provider "${provider}". Supported: "watsonx"`,
      )
  }
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

function combinedProviderError(candidates: Candidate[]): Error {
  const detail = candidates
    .map(entry => `${entry.name}: ${entry.error?.message ?? 'not configured'}`)
    .join('. ')
  return new Error(`LLM providers unavailable. ${detail}.`)
}

class FallbackLLMAdapter implements LLMAdapter {
  /** Name and model of whichever provider last answered, for enrichment provenance. */
  private lastProvider: string | undefined
  private lastModelId: string | undefined

  constructor(private readonly candidates: Candidate[]) {}

  /**
   * Provider recorded on enrichments: the one that actually answered, or the chain's
   * own name before any call has succeeded.
   */
  get provider(): string {
    return this.lastProvider ?? 'watsonx-with-openai-fallback'
  }

  /**
   * Model id recorded on enrichments, or `undefined` until a call succeeds.
   *
   * Deliberately does NOT fall back to the first *constructed* provider's model: a
   * provider can construct (its key is present) and still never answer (the key is
   * rejected), and naming it would record a model that produced none of the text.
   * Callers read this after their calls complete; 'unknown' beats a false attribution.
   */
  get modelId(): string | undefined {
    return this.lastModelId
  }

  summarizeModule(context: ModuleContext): Promise<string> {
    return this.call('summarizeModule', adapter => adapter.summarizeModule(context))
  }

  classifyLayer(context: ModuleContext): Promise<string> {
    return this.call('classifyLayer', adapter => adapter.classifyLayer(context))
  }

  explainDiff(context: DiffContext): Promise<string> {
    return this.call('explainDiff', adapter => adapter.explainDiff(context))
  }

  answerQuestion(context: QueryContext): Promise<string> {
    return this.call('answerQuestion', adapter => adapter.answerQuestion(context))
  }

  async describeModule(context: ModuleContext, preamble?: string): Promise<ModuleDescription> {
    return this.call('describeModule', async adapter => {
      if (!adapter.describeModule) throw new Error('describeModule is not supported')
      return adapter.describeModule(context, preamble)
    })
  }

  getUsage(): TokenUsage | undefined {
    const usages = this.candidates
      .map(entry => entry.adapter?.getUsage?.())
      .filter((usage): usage is TokenUsage => usage !== undefined)
    if (usages.length === 0) return undefined
    return usages.reduce(
      (total, usage) => ({
        promptTokens: total.promptTokens + usage.promptTokens,
        completionTokens: total.completionTokens + usage.completionTokens,
        totalTokens: total.totalTokens + usage.totalTokens,
        callCount: total.callCount + usage.callCount,
      }),
      { promptTokens: 0, completionTokens: 0, totalTokens: 0, callCount: 0 },
    )
  }

  private async call<T>(
    operation: string,
    invoke: (adapter: LLMAdapter) => Promise<T>,
  ): Promise<T> {
    const attempts: Candidate[] = []
    for (const entry of this.candidates) {
      if (!entry.adapter) {
        attempts.push(entry)
        continue
      }
      try {
        const result = await invoke(entry.adapter)
        console.warn(`[debob llm] ${entry.name} handled ${operation}`)
        this.lastProvider = (entry.adapter as unknown as { provider?: string }).provider ?? entry.name
        this.lastModelId =
          (entry.adapter as unknown as { modelId?: string }).modelId ?? this.lastModelId
        return result
      } catch (error) {
        attempts.push({ name: entry.name, error: asError(error) })
      }
    }
    throw combinedProviderError(attempts)
  }
}

export type { LLMAdapter, LLMConfig }
export { WatsonxProvider }
export { OpenAIProvider }
export { OpenRouterProvider }
// Backward-compat alias
export { WatsonxProvider as WatsonxAdapter }
