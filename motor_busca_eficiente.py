# Desafio de aprendizagem: motor de busca eficiente
# Combina os conceitos das práticas 1, 2 e 3:
# - índice direto e invertido
# - pipeline de pré-processamento
# - cálculo de TF-IDF

import math
import re
import textwrap
import unicodedata
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import streamlit as st


STOPWORDS = {
    "a", "as", "o", "os", "e", "de", "do", "da", "dos", "das", "para",
    "em", "com", "sem", "por", "na", "no", "nas", "nos", "mais", "menos",
    "como", "que", "da", "do", "das", "dos", "um", "uma", "uns", "umas",
    "é", "foi", "sao", "ser", "se", "ao", "aos", "ou", "mas", "onde", "quando",
    "tambem", "entao", "logo", "pois", "entre", "sobre", "durante", "para",
    "com", "sem", "para", "apenas", "ideal", "grande", "constante", "melhora",
}

DOCS = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]


def normalize(text):
    text = text.lower()
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")


def tokenize(text):
    return re.findall(r"\b\w+\b", text)


def stemmer_simples(palavra):
    if palavra.endswith("amentos"):
        return palavra[:-4]
    if palavra.endswith("adoras"):
        return palavra[:-3]
    if palavra.endswith("ação"):
        return palavra[:-2]
    if palavra.endswith("ações"):
        return palavra[:-3]
    if palavra.endswith("mente"):
        return palavra[:-4]
    if palavra.endswith("agem"):
        return palavra[:-3]
    if palavra.endswith("oso"):
        return palavra[:-2]
    if palavra.endswith("osa"):
        return palavra[:-2]
    if palavra.endswith("eiro"):
        return palavra[:-3]
    if palavra.endswith("eira"):
        return palavra[:-3]
    if palavra.endswith("ados"):
        return palavra[:-3]
    if palavra.endswith("adas"):
        return palavra[:-3]
    if palavra.endswith("amento"):
        return palavra[:-4]
    if palavra.endswith("a") and len(palavra) > 4:
        return palavra[:-1]
    if palavra.endswith("s") and len(palavra) > 4:
        return palavra[:-1]
    return palavra


def preprocess(text, use_stopwords=True, use_stemming=True):
    tokens = tokenize(text)
    norm = [normalize(t) for t in tokens]
    if use_stopwords:
        filtered = [t for t in norm if t not in STOPWORDS]
    else:
        filtered = norm
    stemmed = [stemmer_simples(t) for t in filtered] if use_stemming else filtered
    return {
        "tokens": tokens,
        "normalizados": norm,
        "sem_stopwords": filtered,
        "stemmed": stemmed,
    }


def build_index(docs, use_stopwords=True, use_stemming=True):
    indice_direto = {}
    indice_invertido = defaultdict(list)
    pipeline_docs = {}

    for i, doc in enumerate(docs, start=1):
        etapas = preprocess(doc, use_stopwords=use_stopwords, use_stemming=use_stemming)
        indice_direto[f"Doc{i}"] = etapas["stemmed"]
        pipeline_docs[f"Doc{i}"] = etapas

    for doc_id, termos in indice_direto.items():
        for termo in sorted(set(termos)):
            if doc_id not in indice_invertido[termo]:
                indice_invertido[termo].append(doc_id)

    return indice_direto, dict(indice_invertido), pipeline_docs


def calcular_ranking(docs, query, use_stopwords=True, use_stemming=True):
    indice_direto, indice_invertido, pipeline_docs = build_index(docs, use_stopwords, use_stemming)
    query_terms = preprocess(query, use_stopwords=use_stopwords, use_stemming=use_stemming)["stemmed"]
    query_terms = sorted(set(query_terms))
    N = len(docs)

    if not query_terms:
        return [], indice_direto, indice_invertido, pipeline_docs

    df = {termo: sum(1 for termos in indice_direto.values() if termo in termos) for termo in query_terms}
    idf = {termo: math.log(N / df[termo]) if df[termo] > 0 else 0.0 for termo in query_terms}

    resultados = []
    for doc_id, termos in indice_direto.items():
        score = 0.0
        detalhes = []
        for termo in query_terms:
            tf = termos.count(termo) / len(termos) if termos else 0.0
            tfidf = tf * idf[termo]
            score += tfidf
            detalhes.append({
                "termo": termo,
                "TF": round(tf, 4),
                "IDF": round(idf[termo], 4),
                "TF-IDF": round(tfidf, 4),
            })
        resultados.append({
            "Documento": doc_id,
            "Score": round(score, 6),
            "Detalhes": detalhes,
        })

    resultados.sort(key=lambda item: item["Score"], reverse=True)
    return resultados, indice_direto, indice_invertido, pipeline_docs


def calcular_cosseno(docs, query, use_stopwords=True, use_stemming=True):
    indice_direto, _, _ = build_index(docs, use_stopwords, use_stemming)
    query_terms = preprocess(query, use_stopwords=use_stopwords, use_stemming=use_stemming)["stemmed"]
    if not query_terms:
        return {}

    vocab = sorted(set(term for termos in indice_direto.values() for term in termos) | set(query_terms))
    df = {term: sum(1 for termos in indice_direto.values() if term in termos) for term in vocab}
    idf = {term: math.log(len(docs) / df[term]) if df[term] > 0 else 0.0 for term in vocab}

    qv = {term: query_terms.count(term) / len(query_terms) * idf.get(term, 0.0) for term in vocab}
    scores = {}
    for doc_id, termos in indice_direto.items():
        dv = {term: termos.count(term) / len(termos) * idf.get(term, 0.0) for term in vocab if term in termos}
        dot = sum(qv.get(term, 0.0) * dv.get(term, 0.0) for term in vocab)
        qnorm = math.sqrt(sum(v * v for v in qv.values()))
        dnorm = math.sqrt(sum(v * v for v in dv.values()))
        similarity = dot / (qnorm * dnorm) if qnorm and dnorm else 0.0
        scores[doc_id] = round(similarity, 6)
    return scores


def escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_bytes(lines):
    page_width = 595
    page_height = 842
    margin = 44
    content_top = 726
    content_bottom = 58
    rendered_lines = []

    for line in lines:
        if not line:
            rendered_lines.append(("space", ""))
            continue
        if re.match(r"^\d+\. ", line):
            rendered_lines.append(("section", line))
            continue
        wrapped = textwrap.wrap(line, width=96, break_long_words=False, break_on_hyphens=False)
        rendered_lines.extend(("body", part) for part in wrapped or [""])

    pages = []
    current_page = []
    y = content_top
    for kind, line in rendered_lines:
        line_height = 18 if kind == "section" else 11 if kind == "body" else 5
        if y - line_height < content_bottom:
            pages.append(current_page)
            current_page = []
            y = content_top
        current_page.append((kind, line, y))
        y -= line_height
    if current_page:
        pages.append(current_page)

    page_contents = []
    for page_number, page in enumerate(pages, start=1):
        stream_lines = [
            "q 0.97 0.98 0.96 rg 0 0 595 842 re f Q",
            "q 0.10 0.29 0.22 rg 0 760 595 82 re f Q",
            "q 0.87 0.64 0.18 rg 0 754 595 6 re f Q",
            "q 0.95 0.98 0.94 rg BT /F2 23 Tf 44 801 Td (AgroSearch) Tj ET Q",
            "q 0.95 0.98 0.94 rg BT /F1 10 Tf 44 780 Td (Relatorio tecnico | Recuperacao de Informacao e PLN) Tj ET Q",
        ]
        for kind, line, line_y in page:
            if kind == "section":
                stream_lines.extend([
                    f"q 0.82 0.92 0.86 rg {margin - 8} {line_y - 13} 515 20 re f Q",
                    f"BT /F2 11 Tf 50 {line_y - 7} Td ({escape_pdf_text(line)}) Tj ET",
                ])
            elif kind == "body":
                stream_lines.append(
                    f"BT /F1 8 Tf {margin} {line_y - 5} Td ({escape_pdf_text(line)}) Tj ET"
                )
            else:
                stream_lines.append(
                    f"q 0.80 0.84 0.80 rg {margin} {line_y - 3} 500 1 re f Q"
                )
        stream_lines.extend([
            f"BT /F1 8 Tf 44  thirty Td (AgroSearch | Entrega integradora) Tj ET".replace("thirty", "30"),
            f"BT /F1 8 Tf 510 30 Td ({page_number}/{len(pages)}) Tj ET",
        ])
        page_contents.append("\n".join(stream_lines).encode("cp1252", "replace"))

    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"

    page_entries = []
    page_content_objects = []
    for i, content in enumerate(page_contents):
        page_obj_number = 6 + (i * 2)
        content_obj_number = page_obj_number + 1
        page_entries.append(f"{page_obj_number} 0 R")
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] /Contents {content_obj_number} 0 R "
            f"/Resources << /Font << /F1 3 0 R /F2 5 0 R >> >> >>"
        ).encode("latin-1")
        content_obj = f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
        page_content_objects.extend([page_obj, content_obj])

    pages = (
        f"<< /Type /Pages /Kids [{' '.join(page_entries)}] /Count {len(page_entries)} >>"
    ).encode("latin-1")

    bold_font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objects = [catalog, pages, font, b"", bold_font] + page_content_objects

    output = [b"%PDF-1.4\n"]
    offsets = [0]
    for obj in objects:
        offsets.append(len(b"".join(output)))
        output.append(f"{len(offsets)-1} 0 obj\n".encode("latin-1"))
        output.append(obj)
        output.append(b"\nendobj\n")

    xref_pos = len(b"".join(output))
    output.append(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    output.append(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    output.append(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1"))
    return b"".join(output)


def gerar_pdf_bytes(docs, query, use_stopwords=True, use_stemming=True):
    indice_direto, indice_invertido, pipeline_docs = build_index(docs, use_stopwords, use_stemming)
    resultados, _, _, _ = calcular_ranking(docs, query, use_stopwords, use_stemming)

    lines = [
        "AgroSearch - Motor de Busca Inteligente",
        "",
        "Desafio Integrador - Recuperação de Informação / PLN",
        "",
        "1. Base de documentos",
    ]
    for i, doc in enumerate(docs, start=1):
        lines.append(f"Doc{i}: {doc}")

    lines.extend(["", "2. Pipeline de pré-processamento"])
    for i, doc in enumerate(docs, start=1):
        etapas = pipeline_docs[f"Doc{i}"]
        lines.append(f"Doc{i}: Tokens={etapas['tokens']}")
        lines.append(f"Doc{i}: Normalizados={etapas['normalizados']}")
        lines.append(f"Doc{i}: Sem stopwords={etapas['sem_stopwords']}")
        lines.append(f"Doc{i}: Stemmed={etapas['stemmed']}")

    lines.extend(["", "3. Índice invertido"])
    for termo, ids in indice_invertido.items():
        lines.append(f"{termo}: {ids}")

    lines.extend(["", "4. Consulta e ranqueamento TF-IDF"])
    lines.append(f"Query: {query}")
    for item in resultados:
        lines.append(f"{item['Documento']}: Score={item['Score']}")

    lines.extend([
        "",
        "5. Conclusão",
        "O índice invertido acelera a busca por termo e o TF-IDF ordena os documentos",
        "por relevância, permitindo ao técnico localizar rapidamente as informações mais úteis.",
    ])

    return build_pdf_bytes(lines)


def gerar_pdf(caminho_saida, docs, query, use_stopwords=True, use_stemming=True):
    pdf_bytes = gerar_pdf_bytes(docs, query, use_stopwords, use_stemming)
    caminho = Path(caminho_saida)
    caminho.write_bytes(pdf_bytes)
    return caminho


def main():
    st.set_page_config(page_title="AgroSearch", page_icon="🌾", layout="wide")
    st.title("🌾 AgroSearch — Motor de Busca Inteligente")
    st.markdown(
        """
        Este protótipo foi desenvolvido para demonstrar como um motor de busca textual pode combinar
        pré-processamento, índice invertido e TF-IDF para localizar os documentos mais relevantes para uma consulta.
        """
    )

    with st.sidebar:
        st.header("Configurações da busca")
        usar_stopwords = st.checkbox("Remover Stopwords", value=True)
        usar_stemming = st.checkbox("Aplicar Stemming", value=True)
        st.caption("Entrega prevista pelo desafio: um arquivo .py e um relatório em PDF breve, com foco na funcionalidade e no entendimento do mecanismo de busca.")

        if st.button("Gerar relatório em PDF"):
            pdf_bytes = gerar_pdf_bytes(DOCS, st.session_state.get("query", "soja"), usar_stopwords, usar_stemming)
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_name"] = "agrosearch_relatorio.pdf"
            st.success("Relatório gerado com sucesso.")
            with st.sidebar:
                st.markdown("### Download do relatório")
                st.caption("Clique abaixo para salvar o PDF na pasta de sua escolha.")
                st.download_button(
                    label="Baixar relatório em PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=st.session_state["pdf_name"],
                    mime="application/pdf",
                )

    query = st.text_input("Consulta (Query):", value="soja")
    st.session_state["query"] = query

    if query:
        resultados, indice_direto, indice_invertido, pipeline_docs = calcular_ranking(DOCS, query, usar_stopwords, usar_stemming)
        cosseno = calcular_cosseno(DOCS, query, usar_stopwords, usar_stemming)

        st.info("A busca foi processada com as etapas de tokenização, normalização, remoção de stopwords e stemming, conforme o pipeline do desafio.")

        with st.expander("1. Pipeline de pré-processamento", expanded=True):
            for doc_id, etapas in pipeline_docs.items():
                st.markdown(f"### {doc_id}")
                st.json({
                    "Tokens": etapas["tokens"],
                    "Normalizados": etapas["normalizados"],
                    "Sem Stopwords": etapas["sem_stopwords"],
                    "Stemmed": etapas["stemmed"],
                })

        with st.expander("2. Índice Invertido"):
            st.json(indice_invertido)

        with st.expander("3. Busca + Ranqueamento TF-IDF"):
            if resultados:
                tabela = []
                for item in resultados:
                    tabela.append({
                        "Documento": item["Documento"],
                        "Score TF-IDF": item["Score"],
                        "Similaridade cosseno": cosseno.get(item["Documento"], 0.0),
                    })
                linhas_tabela = [
                    "| Documento | Score TF-IDF | Similaridade cosseno |",
                    "|:--|--:|--:|",
                ]
                linhas_tabela.extend(
                    f"| {linha['Documento']} | {linha['Score TF-IDF']:.6f} | {linha['Similaridade cosseno']:.6f} |"
                    for linha in tabela
                )
                st.markdown("\n".join(linhas_tabela))
                vencedor = resultados[0]
                st.success(f"Documento mais relevante: {vencedor['Documento']} com score {vencedor['Score']}")

                for item in resultados:
                    st.markdown(f"### {item['Documento']}")
                    st.json(item["Detalhes"])
            else:
                st.info("A consulta não retornou termos relevantes após o pré-processamento.")

        st.caption("Observação: o documento do desafio exige 4 etapas no pipeline, índice invertido e TF-IDF implementados do zero, sem bibliotecas de alto nível.")
    else:
        st.warning("Digite uma consulta para executar a busca.")


if __name__ == "__main__":
    main()
