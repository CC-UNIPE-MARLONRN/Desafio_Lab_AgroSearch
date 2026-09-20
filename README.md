# AgroSearch

Motor de busca textual para o desafio integrador de Recuperacao de Informacao / PLN.

## Requisitos atendidos

- Pipeline com tokenizacao, normalizacao, stopwords e stemming.
- Indice direto e indice invertido em memoria.
- Calculo de TF, IDF e TF-IDF implementado do zero.
- Interface interativa em Streamlit com controles de stopwords e stemming.
- Ranking por TF-IDF e similaridade de cosseno como bonus.
- Geracao de relatorio em PDF pela propria aplicacao.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run motor_busca_eficiente.py
```

O aplicativo sera aberto pelo Streamlit no endereco local informado no terminal.

## Estrutura da entrega

- `motor_busca_eficiente.py`: aplicacao Streamlit completa.
- `agrosearch_relatorio.pdf`: relatorio breve da entrega.
- `requirements.txt`: dependencia necessaria para executar a aplicacao.
