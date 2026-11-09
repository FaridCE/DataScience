---
layout: post
title: "Local RAG Web Streaming Interface"
date: 2026-11-09
categories: rag, llm
---

# Asistente RAG Local (Streaming Interface)

Esta publicación integra el cliente web ejecutable para conectar GitHub Pages con tu pipeline RAG local (**PostgreSQL + pgvector + Ollama**) utilizando un túnel seguro **HTTPS**.

<!-- Embedded Streaming Interface -->
<div id="rag-container" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin: 24px 0; font-family: system-ui, -apple-system, sans-serif; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
    <h3 style="margin-top: 0; color: #1e3a8a;">Consulta al Dictamen Jurídico RAG</h3>
    
    <div id="chat-box" style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 16px; min-height: 200px; max-height: 450px; overflow-y: auto; font-family: monospace; font-size: 14px; white-space: pre-wrap; margin-bottom: 16px; color: #0f172a;">Escribe tu pregunta sobre legislación financiera o jurisprudencia SCJN...</div>

    <div style="display: flex; gap: 8px;">
        <input type="text" id="prompt-input" placeholder="Ej. ¿Qué regulación aplica a las ITF?" style="flex: 1; padding: 10px 14px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; outline: none;">
        <button id="submit-btn" onclick="sendPrompt()" style="padding: 10px 20px; background: #2563eb; color: #ffffff; border: none; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer;">Enviar</button>
    </div>
</div>

{% raw %}
<script>
    // Reemplaza con tu URL activa de Cloudflare Tunnel o Ngrok
    const TUNNEL_URL = "https://excitement-aids-guru-holding.trycloudflare.com";

    async function sendPrompt() {
        const inputEl = document.getElementById("prompt-input");
        const box = document.getElementById("chat-box");
        const btn = document.getElementById("submit-btn");
        const question = inputEl.value.trim();

        if (!question) return;

        // UI state management
        box.innerText = "Buscando vectores relevantes y generando respuesta en tiempo real...\n\n";
        inputEl.value = "";
        btn.disabled = true;

        try {
            const response = await fetch(`${TUNNEL_URL}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                throw new Error(`HTTP Error status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // Retener chunk incompleto en el buffer

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.replace("data: ", "").trim();
                        if (dataStr === "[DONE]") break;
                        try {
                            const parsed = JSON.parse(dataStr);
                            box.innerText += parsed.token;
                            box.scrollTop = box.scrollHeight;
                        } catch (e) {}
                    }
                }
            }
        } catch (err) {
            box.innerText += "\n\n[Error de conexión: Asegúrate de que el backend FastAPI local y el túnel HTTPS estén activos.]";
            console.error("Error en streaming:", err);
        } finally {
            btn.disabled = false;
        }
    }

    // Permitir envío presionando la tecla Enter
    document.addEventListener("DOMContentLoaded", () => {
        const inputEl = document.getElementById("prompt-input");
        if (inputEl) {
            inputEl.addEventListener("keypress", (e) => {
                if (e.key === "Enter") {
                    sendPrompt();
                }
            });
        }
    });
</script>
{% endraw %}

---

## Guía de Configuración Rápida

1. **Backend FastAPI**: Asegúrate de que tu script `main.py` esté ejecutándose en `http://localhost:8000`.
2. **Túnel HTTPS**: Abre una terminal y expón el puerto 8000 mediante Cloudflare Tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8000