import type { LLMAdapter } from './adapter.js'
import type { LLMConfig } from './adapter.js'
import { WatsonxProvider } from './providers/watsonx.js'
import { OpenRouterProvider } from './providers/openrouter.js'
import type { DiffContext, ModuleContext, ModuleDescription, QueryContext, TokenUsage } from './adapter.js'

// ─── LLM Adapter Factory ──────────────────────────────────────────────────────

/**
 * Create an LLMAdapter for the given provider.
 *
 * V1 supported providers:
 *  - `"watsonx"` → `WatsonxProvider` (IBM watsonx.ai SDK, chat API)
 *
 * Future providers are added by importing their class and extending the switch.
 *
 * @throws If `provider` is not a recognised value.
 */
export function createLLMAdapter(provider: string, config: LLMConfig): LLMAdapter {
  switch (provider) {
    case 'watsonx': {
      let watsonx: LLMAdapter | undefined
      let watsonxError: Error | undefined
      try {
        watsonx = new WatsonxProvider(config)
      } catch (error) {
        watsonxError = asError(error)
      }

      let openrouter: LLMAdapter | undefined
      let openrouterError: Error | undefined
      try {
        openrouter = new OpenRouterProvider()
      } catch (error) {
        openrouterError = asError(error)
      }

      if (!watsonx && !openrouter) {
        throw combinedProviderError(watsonxError, openrouterError)
      }
      return new FallbackLLMAdapter(watsonx, openrouter, watsonxError, openrouterError)
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

function combinedProviderError(watsonxError?: Error, openrouterError?: Error): Error {
  return new Error(
    `LLM providers unavailable. Watsonx: ${watsonxError?.message ?? 'not configured'}. ` +
      `OpenRouter: ${openrouterError?.message ?? 'not configured'}.`,
  )
}

class FallbackLLMAdapter implements LLMAdapter {
  readonly provider = 'watsonx-with-openrouter-fallback'

  constructor(
    private readonly watsonx: LLMAdapter | undefined,
    private readonly openrouter: LLMAdapter | undefined,
    private readonly watsonxConstructionError?: Error,
    private readonly openrouterConstructionError?: Error,
  ) {}

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
    const usages = [this.watsonx?.getUsage?.(), this.openrouter?.getUsage?.()].filter(
      (usage): usage is TokenUsage => usage !== undefined,
    )
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
    let watsonxError = this.watsonxConstructionError
    if (this.watsonx) {
      try {
        const result = await invoke(this.watsonx)
        console.warn(`[debob llm] watsonx handled ${operation}`)
        return result
      } catch (error) {
        watsonxError = asError(error)
      }
    }

    let openrouterError = this.openrouterConstructionError
    if (this.openrouter) {
      try {
        const result = await invoke(this.openrouter)
        console.warn(`[debob llm] OpenRouter handled ${operation}`)
        return result
      } catch (error) {
        openrouterError = asError(error)
      }
    }

    throw combinedProviderError(watsonxError, openrouterError)
  }
}

export type { LLMAdapter, LLMConfig }
export { WatsonxProvider }
export { OpenRouterProvider }
// Backward-compat alias
export { WatsonxProvider as WatsonxAdapter }
