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
    const TUNNEL_URL = "https://defend-jack-opponents-declined.trycloudflare.com";

    async function sendPrompt() {
        const inputEl = document.getElementById("prompt-input");
        const box = document.getElementById("chat-box");
        const btn = document.getElementById("submit-btn");
        const pdfArea = document.getElementById("pdf-action-area");
        const question = inputEl.value.trim();

        if (!question) return;

        // Reset UI components
        box.innerText = "🔍 Buscando legislación y precedentes SCJN...\n\n";
        pdfArea.innerHTML = "";
        inputEl.value = "";
        btn.disabled = true;

        try {
            const response = await fetch(`${TUNNEL_URL}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                throw new Error(`HTTP Status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.replace("data: ", "").trim();
                        if (dataStr === "[DONE]") break;
                        try {
                            const parsed = JSON.parse(dataStr);

                            // Append LLM tokens to chat window
                            if (parsed.token) {
                                box.innerText += parsed.token;
                                box.scrollTop = box.scrollHeight;
                            }

                            // Triggered when PDF compilation completes on backend
                            if (parsed.pdf_url) {
                                const fullPdfUrl = `${TUNNEL_URL}${parsed.pdf_url}`;
                                renderPdfControls(fullPdfUrl);
                            }
                        } catch (e) {}
                    }
                }
            }
        } catch (err) {
            box.innerText += `\n\n[Error de conexión: ${err.message}]`;
        } finally {
            btn.disabled = false;
        }
    }

    function renderPdfControls(pdfUrl) {
        const pdfArea = document.getElementById("pdf-action-area");
        pdfArea.innerHTML = `
            <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <div style="font-weight: 600; color: #0369a1; margin-bottom: 8px;">📄 Dictamen PDF Compilado Exitosamente</div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <a href="${pdfUrl}" target="_blank" download style="padding: 8px 16px; background: #0284c7; color: white; border-radius: 4px; text-decoration: none; font-weight: 600; font-size: 14px;">
                        📥 Descargar PDF
                    </a>
                    <button onclick="togglePdfViewer('${pdfUrl}')" style="padding: 8px 16px; background: #334155; color: white; border: none; border-radius: 4px; font-weight: 600; font-size: 14px; cursor: pointer;">
                        👁️ Ver PDF Integrado
                    </button>
                </div>
                <div id="pdf-viewer-frame" style="display: none; margin-top: 14px;">
                    <iframe src="${pdfUrl}" width="100%" height="550px" style="border: 1px solid #cbd5e1; border-radius: 6px;"></iframe>
                </div>
            </div>
        `;
    }

    function togglePdfViewer(pdfUrl) {
        const frame = document.getElementById("pdf-viewer-frame");
        if (frame.style.display === "none") {
            frame.style.display = "block";
        } else {
            frame.style.display = "none";
        }
    }
</script>
{% endraw %}

---