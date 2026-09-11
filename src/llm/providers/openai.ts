import type { LLMConfig } from '../adapter.js'
import { OpenAICompatibleProvider } from './openai-compatible.js'

const OPENAI_URL = 'https://api.openai.com/v1/chat/completions'
const DEFAULT_MODEL = 'gpt-4o-mini'

/**
 * OpenAI-backed adapter, talking to `api.openai.com` directly.
 *
 * Requires an OpenAI platform key (`sk-proj-…` / `sk-…`) in `OPENAI_API_KEY` — an
 * OpenRouter key will not authenticate here, and vice versa. Model is chosen with
 * `OPENAI_MODEL`, defaulting to `gpt-4o-mini`: cheap, fast, and reliable at the
 * JSON-only replies `describeModule` needs to avoid falling back to two calls.
 */
export class OpenAIProvider extends OpenAICompatibleProvider {
  constructor(_config?: LLMConfig) {
    const apiKey = process.env['OPENAI_API_KEY']
    if (!apiKey) throw new Error('OpenAIProvider: OPENAI_API_KEY is required')
    super({
      provider: 'openai',
      endpoint: OPENAI_URL,
      apiKey,
      model: process.env['OPENAI_MODEL'] || DEFAULT_MODEL,
      label: 'OpenAI',
    })
  }
}

export { DEFAULT_MODEL as OPENAI_DEFAULT_MODEL }
