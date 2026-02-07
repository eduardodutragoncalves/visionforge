const ENGINE_HOST = import.meta.env.VITE_ENGINE_HOST ?? '127.0.0.1';
const ENGINE_PORT = import.meta.env.VITE_ENGINE_PORT ?? '8000';

const DEFAULT_HTTP = `http://${ENGINE_HOST}:${ENGINE_PORT}`;
const DEFAULT_WS = `ws://${ENGINE_HOST}:${ENGINE_PORT}/ws`;

export const ENGINE_HTTP_URL = import.meta.env.VITE_ENGINE_URL ?? DEFAULT_HTTP;
export const ENGINE_WS_URL = import.meta.env.VITE_ENGINE_WS_URL ?? DEFAULT_WS;
