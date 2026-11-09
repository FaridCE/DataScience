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
        <button id="submit-btn" onclick="sendPrompt()" style="padding: 10px