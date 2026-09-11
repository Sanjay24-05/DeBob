import type { LLMConfig } from '../adapter.js'
import { OpenAICompatibleProvider } from './openai-compatible.js'

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
const DEFAULT_MODEL = 'google/gemma-4-31b-it:free'

/**
 * OpenRouter-backed adapter — the fallback provider when watsonx is unavailable.
 *
 * Model is chosen with `OPENROUTER_MODEL`. Prefer a non-reasoning model: reasoning
 * models spend most of their completion budget on hidden `reasoning_content`, which
 * makes enrichment slow without improving the summaries, and `:free` models are
 * queued behind paid traffic.
 */
export class OpenRouterProvider extends OpenAICompatibleProvider {
  constructor(_config?: LLMConfig) {
    const apiKey = process.env['OPENROUTER_API_KEY']
    if (!apiKey) throw new Error('OpenRouterProvider: OPENROUTER_API_KEY is required')
    super({
      provider: 'openrouter',
      endpoint: OPENROUTER_URL,
      apiKey,
      model: process.env['OPENROUTER_MODEL'] || DEFAULT_MODEL,
      label: 'OpenRouter',
    })
  }
}

export { DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL }
