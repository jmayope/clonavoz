import torch
from TTS.api import TTS

# Detectar si existe una GPU NVIDIA compatible
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Utilizando dispositivo: {device}")

# Cargar modelo XTTS-v2
tts = TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2"
).to(device)

# Audio de referencia con tu voz
speaker_audio = "mi_voz.wav"

# Texto que deseas reproducir
texto = """
Después de desarrollar los cinco casos, podemos observar que cada metodología responde a una necesidad diferente.
La matriz FODA nos permite evaluar las condiciones internas y externas antes de iniciar un proyecto.
La metodología de problemas duros permite establecer objetivos cuantificables y verificar resultados.
Hall y Jenkins nos ayuda a estructurar una intervención de ingeniería, desde la identificación del problema hasta la implantación.
La MSS permite comprender diferentes perspectivas y construir soluciones que sean aceptadas por las personas involucradas.
Finalmente, Wilson permite organizar la información, identificar responsables y establecer relaciones entre actividades y procesos.
Por lo tanto, no existe una metodología única que resuelva todos los problemas. Su elección depende de las características de cada organización y de la naturaleza del problema.
Para finalizar, quiero destacar cinco conclusiones de este trabajo.
Primero, una automatización exitosa comienza definiendo correctamente el problema y no eligiendo directamente la herramienta tecnológica.
Segundo, los problemas duros necesitan indicadores y resultados medibles, mientras que los problemas suaves requieren escuchar a las personas y comprender sus diferentes perspectivas.
Tercero, debemos entender a las organizaciones como sistemas abiertos, donde una modificación puede afectar diferentes áreas, procesos y personas.
Cuarto, la automatización requiere combinar conocimientos técnicos con gestión, seguridad, comunicación y comprensión del negocio.
Y quinto, las metodologías sistémicas no son recetas rígidas, sino herramientas que nos permiten analizar situaciones y tomar mejores decisiones.
Como reflexión final, considero que n8n es una herramienta que facilita la integración y automatización de procesos, pero no soluciona por sí sola los problemas de una organización.
Para conseguir resultados realmente efectivos, primero debemos comprender el sistema, identificar sus necesidades, involucrar a las personas y establecer controles adecuados.
"""

# Generar audio utilizando tu voz
tts.tts_to_file(
    text=texto,
    speaker_wav=speaker_audio,
    language="es",
    file_path="mi_voz_generada.wav",
    split_sentences=True
)

print("Audio generado correctamente.")