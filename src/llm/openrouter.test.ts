import { afterEach, describe, expect, it, vi } from 'vitest'
import { createLLMAdapter } from './index.js'
import { OpenRouterProvider } from './providers/openrouter.js'

const originalApiKey = process.env['OPENROUTER_API_KEY']
const originalModel = process.env['OPENROUTER_MODEL']
const originalWatsonxApiKey = process.env['WATSONX_API_KEY']
const originalWatsonxProjectId = process.env['WATSONX_PROJECT_ID']
const originalWatsonxUrl = process.env['WATSONX_URL']
const originalWatsonxModelId = process.env['WATSONX_MODEL_ID']
const originalFetch = globalThis.fetch

const context = {
  filePath: 'src/example.ts',
  imports: [],
  reExports: [],
  declarations: [],
}

function restoreEnvironment(): void {
  if (originalApiKey === undefined) delete process.env['OPENROUTER_API_KEY']
  else process.env['OPENROUTER_API_KEY'] = originalApiKey
  if (originalModel === undefined) delete process.env['OPENROUTER_MODEL']
  else process.env['OPENROUTER_MODEL'] = originalModel
  if (originalWatsonxApiKey === undefined) delete process.env['WATSONX_API_KEY']
  else process.env['WATSONX_API_KEY'] = originalWatsonxApiKey
  if (originalWatsonxProjectId === undefined) delete process.env['WATSONX_PROJECT_ID']
  else process.env['WATSONX_PROJECT_ID'] = originalWatsonxProjectId
  if (originalWatsonxUrl === undefined) delete process.env['WATSONX_URL']
  else process.env['WATSONX_URL'] = originalWatsonxUrl
  if (originalWatsonxModelId === undefined) delete process.env['WATSONX_MODEL_ID']
  else process.env['WATSONX_MODEL_ID'] = originalWatsonxModelId
  globalThis.fetch = originalFetch
}

afterEach(() => {
  vi.restoreAllMocks()
  restoreEnvironment()
})

describe('OpenRouterProvider', () => {
  it('sends the required chat-completions request', async () => {
    process.env['OPENROUTER_API_KEY'] = 'test-key'
    process.env['OPENROUTER_MODEL'] = 'test/model'
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: 'summary' } }] }), { status: 200 }),
    )
    globalThis.fetch = fetchMock

    const result = await new OpenRouterProvider().summarizeModule(context)

    expect(result).toBe('summary')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('https://openrouter.ai/api/v1/chat/completions')
    expect(init?.method).toBe('POST')
    expect(init?.headers).toEqual({
      Authorization: 'Bearer test-key',
      'Content-Type': 'application/json',
    })
    expect(JSON.parse(String(init?.body))).toMatchObject({
      model: 'test/model',
      messages: [{ role: 'user', content: expect.stringContaining('src/example.ts') }],
    })
  })

  it('requires OPENROUTER_API_KEY', () => {
    delete process.env['OPENROUTER_API_KEY']
    expect(() => new OpenRouterProvider()).toThrow('OPENROUTER_API_KEY')
  })

  it('describes rate limits with the provider and status', async () => {
    process.env['OPENROUTER_API_KEY'] = 'test-key'
    globalThis.fetch = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('slow down', { status: 429, statusText: 'Too Many Requests' }),
    )

    await expect(new OpenRouterProvider().summarizeModule(context)).rejects.toThrow(
      'OpenRouter request failed with HTTP 429 (rate limit exceeded)',
    )
  })
})

describe('LLM provider fallback', () => {
  it('uses OpenRouter when watsonx credentials are missing', async () => {
    delete process.env['WATSONX_API_KEY']
    delete process.env['WATSONX_PROJECT_ID']
    delete process.env['WATSONX_URL']
    delete process.env['WATSONX_MODEL_ID']
    process.env['OPENROUTER_API_KEY'] = 'test-key'
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: 'fallback summary' } }] }), { status: 200 }),
    )
    globalThis.fetch = fetchMock
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    const adapter = createLLMAdapter('watsonx', {
      provider: 'watsonx',
      apiKey: undefined,
      projectId: undefined,
      url: undefined,
      modelId: undefined,
    })

    await expect(adapter.summarizeModule(context)).resolves.toBe('fallback summary')
    expect(warning).toHaveBeenCalledWith('[debob llm] OpenRouter handled summarizeModule')
  })

  it('reports both provider failures', () => {
    delete process.env['WATSONX_API_KEY']
    delete process.env['WATSONX_PROJECT_ID']
    delete process.env['WATSONX_URL']
    delete process.env['WATSONX_MODEL_ID']
    delete process.env['OPENROUTER_API_KEY']

    expect(() => createLLMAdapter('watsonx', { provider: 'watsonx' })).toThrow(
      /Watsonx:.*WATSONX_API_KEY.*OpenRouter:.*OPENROUTER_API_KEY/,
    )
  })
})
