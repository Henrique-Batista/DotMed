## Agente de IA DotMed - Especificação do Projeto

### 1. Visão Geral
O Agente DotMed é uma solução de inteligência artificial projetada para aliviar a carga do setor de telefonia da clínica, automatizando o suporte a dúvidas sobre preparo de exames.

### 2. Análise de Dores
- **Sobrecarga Telefônica:** Grande volume de chamadas para informações procedimentais (jejum, restrições) impede o foco em agendamentos.
- **Fricção no Atendimento:** Clientes perdem tempo em filas de espera para perguntas de resposta rápida.
- **Consistência da Informação:** Garantir que todos os clientes recebam exatamente as mesmas orientações contidas nos manuais oficiais.

### 3. Arquitetura Proposta (RAG - Retrieval-Augmented Generation)
Para garantir precisão e evitar alucinações, o sistema utilizará a técnica de RAG:
1. **Ingestão:** Leitura dos arquivos em `./contexto` (PDFs e CSV).
2. **Processamento:** Divisão dos textos em chunks e geração de embeddings (Google Embeddings).
3. **Recuperação:** Busca semântica no banco de vetores (FAISS/Chroma) com base na pergunta do usuário.
4. **Geração:** O modelo Gemini processa o contexto recuperado e gera uma resposta gentil e precisa.

### 4. Requisitos Técnicos
- **LLM:** Google Gemini 1.5 Flash (ou superior).
- **Framework:** LangChain para orquestração da cadeia RAG.
- **Interface:** Streamlit (Chat Interface).
- **Base de Conhecimento:** 
  - `manual-de-preparo-para-exames.pdf` (PDF)
  - `Manual-de-preparo-para-realizacao-dos-exames.pdf` (PDF)
  - `planilha-de-procedimentos.csv` (CSV)

### 5. Escopo de Atuação
O agente deve:
- Responder apenas sobre procedimentos médicos e consultas da DotMed.
- Consultar obrigatoriamente a base de dados para instruções de preparo.
- Manter um tom gentil, educado e compreensivo.
- Informar que não pode ajudar com temas fora do escopo médico/clínico.

### 6. Interface de Usuário
- Layout simples com `st.chat_input` e `st.chat_message`.
- Histórico de mensagens persistido durante a sessão.