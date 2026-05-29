# Agente de consultas automatizadas DotMed

Este projeto tem como objetivo implementar o agente de consultas da DotMed, uma clínica e laboratório de exames. Uma grande dor que a DotMed tem é a da comunicação dos clientes com duvidas sobre o andamento dos seus exames e duvidas referentes ao procedimento. Como o setor de telefonia da clinica recebe muitas ligacoes para marcar e desmarcar consultas, foi sugerido o uso de uma infraestrutura de IA que solucionasse este problema, já que a clinica tem bem definido quais os procedimentos e preparacoes devem ser feitos.

## Dependencias

- uv (curl -LsSf https://astral.sh/uv/install.sh | sh)
- streamlit (uv add streamlit)

# Como executar

- Crie o arquivo .env com a variavel de ambiente para a chave de api do Gemini: ``` GEMINI_API_KEY="chave da api"```
- Execute ``` uv run streamlit run main.py ``` e aguarde um pouco para o modelo de embedding local realizar o processo de vetorializacao dos documentos da pasta de contexto. Uma vez feito isto, ele criará uma pasta guardando os embeddings gerados para uso posterior.