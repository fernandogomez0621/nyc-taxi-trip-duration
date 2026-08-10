"""Genera el resumen ejecutivo de una pagina en PDF."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

NEGRO = colors.HexColor("#1A1A1A")
GRIS = colors.HexColor("#4A4A4A")
LINEA = colors.HexColor("#999999")

base = getSampleStyleSheet()

titulo = ParagraphStyle(
    "titulo", parent=base["Normal"], fontName="Times-Bold", fontSize=15.5,
    leading=18, textColor=NEGRO, alignment=TA_CENTER, spaceAfter=3)
subtitulo = ParagraphStyle(
    "subtitulo", parent=base["Normal"], fontName="Times-Italic", fontSize=9.5,
    leading=12, textColor=GRIS, alignment=TA_CENTER, spaceAfter=2)
autor = ParagraphStyle(
    "autor", parent=base["Normal"], fontName="Times-Roman", fontSize=9,
    textColor=NEGRO, alignment=TA_CENTER, spaceAfter=9)
h = ParagraphStyle(
    "h", parent=base["Normal"], fontName="Times-Bold", fontSize=10.5, leading=12,
    textColor=NEGRO, spaceBefore=8, spaceAfter=3)
p = ParagraphStyle(
    "p", parent=base["Normal"], fontName="Times-Roman", fontSize=9.4, leading=11.8,
    textColor=NEGRO, alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=4)
sangria = ParagraphStyle("sangria", parent=p, leftIndent=12, bulletIndent=2, spaceAfter=4)
nota = ParagraphStyle(
    "nota", parent=base["Normal"], fontName="Times-Roman", fontSize=8, leading=9.8,
    textColor=GRIS, alignment=TA_JUSTIFY)

doc = SimpleDocTemplate(
    "/mnt/user-data/outputs/resumen_ejecutivo.pdf", pagesize=A4,
    leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.7 * cm, bottomMargin=1.4 * cm,
    title="Estimacion de la duracion de viajes en taxi - Resumen ejecutivo",
    author="Andres Fernando Gomez Rojas")

S = []
S.append(Paragraph("Estimación de la duración de los viajes en taxi", titulo))
S.append(Paragraph("Hallazgos operativos sobre 1,46 millones de trayectos en Nueva York, "
                   "enero a junio de 2016", subtitulo))
S.append(Paragraph("Andrés Fernando Gómez Rojas", autor))
S.append(HRFlowable(width="100%", thickness=0.8, color=NEGRO, spaceAfter=8))

S.append(Paragraph("Objetivo", h))
S.append(Paragraph(
    "Estimar la duración de un trayecto en el momento en que el pasajero aborda, empleando "
    "únicamente la información disponible en ese instante: punto de recogida, destino, hora y día. "
    "El trabajo entrega un modelo capaz de producir esa estimación y cuatro hallazgos sobre el "
    "comportamiento del tráfico con consecuencias operativas propias.", p))

S.append(Paragraph("Hallazgos", h))
for t_, d in [
    ("La congestión no coincide con la hora punta.",
     "Los taxis circulan más lento entre las 11 de la mañana y las 3 de la tarde (10,4 km/h) que "
     "durante el pico vespertino. El tramo más ágil corresponde a las 5 de la mañana, con 21,2 km/h: "
     "<b>un mismo recorrido llega a tardar el doble según la hora de salida.</b> La demanda de taxis, "
     "en cambio, alcanza su máximo entre las 6 y las 7 de la tarde. Ambos fenómenos no coinciden "
     "porque la congestión depende del tráfico total de la ciudad y no del servicio de taxi."),
    ("El destino pesa el doble que el origen.",
     "A dónde se dirige el pasajero explica cerca del doble de la variación en el tiempo de viaje que "
     "el punto donde se le recoge. El destino determina si el trayecto termina internándose en el "
     "centro o saliendo hacia la periferia."),
    ("La demanda se concentra en pocas zonas.",
     "Tres de las veinte zonas de la ciudad reúnen el 30 % de los viajes, y el 17 % de los trayectos "
     "comienza y termina dentro de la misma zona."),
    ("Los viajes a aeropuerto constituyen un régimen aparte.",
     "Representan el 6,6 % del total, con una duración mediana de 32 minutos frente a 10 del resto. "
     "No obstante, tardan más esencialmente por ser más largos: descontada la distancia, apenas se "
     "comportan de forma distinta."),
]:
    S.append(Paragraph(f"<b>{t_}</b> {d}", sangria, bulletText="—"))

S.append(Paragraph("Desempeño del modelo", h))
S.append(Paragraph(
    "Sobre un trayecto típico de 11 minutos, la estimación se desvía en promedio 3,3 minutos, un "
    "<b>error cercano al 29 %</b>. El criterio adecuado para juzgar esa cifra no es la perfección sino "
    "la alternativa realista. Una regla construida manualmente por un analista —duración habitual "
    "según distancia y hora— se desvía un 34 % más. Una regla que atienda solo al reloj, ignorando la "
    "distancia, resulta <i>peor que responder siempre el promedio</i>.", p))

tabla = Table([
    ["Método de estimación", "Error medio", "Valoración"],
    ["Regla por hora y día de la semana", "7,8 min", "Peor que responder el promedio"],
    ["Regla por distancia y hora", "4,4 min", "Aceptable"],
    ["Modelo propuesto", "3,3 min", "24 % mejor que la regla manual"],
], colWidths=[6.4 * cm, 2.5 * cm, 7.7 * cm])
tabla.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 3), (-1, 3), "Times-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.8),
    ("TEXTCOLOR", (0, 0), (-1, -1), NEGRO),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, NEGRO),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, NEGRO),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, NEGRO),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
]))
S.append(Spacer(1, 3))
S.append(tabla)
S.append(Spacer(1, 6))

S.append(Paragraph(
    "<b>Un matiz que condiciona su uso.</b> El error no se reparte de manera uniforme. El 2 % de los "
    "viajes —aquellos que se demoran de forma excepcional— concentra <b>más de la mitad del error "
    "total</b>. Para el trayecto corriente la estimación es fiable; ante un incidente, un cierre de vía "
    "o un temporal, el modelo queda ciego porque no dispone de esa información. Es el mismo fenómeno "
    "observable el 23 de enero de 2016, cuando la nevada redujo los viajes en un 80 % en un solo día.", p))

S.append(Paragraph("Recomendaciones", h))
for r in [
    "<b>Comunicar una ventana de tiempo en lugar de un minuto exacto</b>, y que sea asimétrica: el "
    "riesgo de demorarse más de lo previsto supera con holgura el de llegar antes. Para un trayecto "
    "típico, anunciar entre 8 y 15 minutos resulta honesto; prometer 11 no lo es.",
    "<b>Reposicionar la flota entre las 3 y las 6 de la madrugada.</b> Es la única franja en que "
    "trasladar vehículos no compite con la congestión, y los desplazamientos consumen la mitad de tiempo.",
    "<b>Concentrar la operación en las zonas de mayor demanda.</b> Tres zonas cubren el 30 % de los "
    "viajes, de modo que una reducción de flota puede diseñarse sin sacrificar cobertura relevante.",
]:
    S.append(Paragraph(r, sangria, bulletText="—"))

S.append(Paragraph("Siguiente paso", h))
S.append(Paragraph(
    "La mayor ganancia disponible no reside en refinar el modelo sino en <b>incorporar datos "
    "meteorológicos y de incidentes de tráfico</b>. Es precisamente la información que explicaría ese "
    "2 % de viajes responsable de la mitad del error. El resto de extensiones —más histórico, eventos "
    "de la ciudad, obras programadas— viene después.", p))

S.append(Spacer(1, 6))
S.append(HRFlowable(width="100%", thickness=0.5, color=LINEA, spaceAfter=5))
S.append(Paragraph(
    "Base analizada: 1.458.644 trayectos registrados por la Comisión de Taxis y Limusinas de Nueva "
    "York. Tras depurar registros inconsistentes —taxímetros que permanecieron abiertos, señales de "
    "posición perdidas— se conservó el 99,8 % de los datos. La evaluación se realizó sobre las dos "
    "últimas semanas del período, no vistas durante el entrenamiento, con el fin de reproducir las "
    "condiciones reales de uso.", nota))

doc.build(S)
print("PDF generado")
