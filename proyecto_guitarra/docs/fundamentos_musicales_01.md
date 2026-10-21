# Fundamentos musicales: primera lección

Los ejemplos siguientes son didácticos y calculados. No son predicciones de un modelo
entrenado. No necesitas saber tocar guitarra para entenderlos.

## 1. ¿Qué queremos recuperar de un sonido?

Una nota tiene una altura: la percibimos como más grave o más aguda. En el proyecto
no basta con reconocer una altura. También queremos estimar en qué cuerda y traste
se produjo y en qué momento estaba sonando.

Una grabación digital es una secuencia de números que describe cómo cambia una señal
de audio con el tiempo. No contiene explícitamente instrucciones como «cuerda 6,
traste 3». La red debe aprender a relacionar patrones del sonido con esas respuestas.

## 2. Frecuencia fundamental y armónicos

La frecuencia cuenta ciclos por segundo y se expresa en hercios (Hz). La frecuencia
fundamental está relacionada con la altura de una nota. Junto con ella aparecen
componentes aproximadamente en múltiplos de esa frecuencia: los armónicos. No son
necesariamente notas adicionales pulsadas por el músico.

Ejemplo idealizado: una fundamental de 110 Hz puede acompañarse de componentes a
220, 330 y 440 Hz. En una guitarra real pueden existir desviaciones. El patrón y
la evolución de esas componentes contribuyen al timbre, la cualidad que permite
distinguir instrumentos y formas de tocar incluso a una misma altura.

Por eso el algoritmo no puede interpretar cada pico de frecuencia como una nota
independiente. Si varias cuerdas suenan, sus componentes pueden superponerse.

## 3. Nombres de notas, semitono y octava

Usaremos C=Do, D=Re, E=Mi, F=Fa, G=Sol, A=La y B=Si. El símbolo # significa
sostenido y b significa bemol. En afinación de temperamento igual, una octava se
divide en doce pasos iguales de razón de frecuencias: los semitonos. Una octava
duplica la frecuencia. Un semitono multiplica por 2^(1/12).

El número en E2 o E4 identifica la octava de la altura que suena; no es una cuerda
ni un traste. En nuestro convenio E2 ronda 82.41 Hz y E4 ronda 329.63 Hz. La partitura
tradicional de guitarra puede escribir una octava por encima del sonido real; aquí
trabajaremos con las alturas sonoras y sus números MIDI para evitar ambigüedades.

## 4. Cuerda al aire y trastes

Una cuerda al aire vibra sin presionar un traste: se representa con el número 0.
Presionar un traste cambia la longitud vibrante y sube la altura. En el modelo ideal
de afinación estándar, cada traste avanza un semitono.

| Cuerda convencional | Sonido al aire | Frecuencia aproximada | Número MIDI |
|---|---|---:|---:|
| 6, la más grave | E2 / Mi2 | 82.41 Hz | 40 |
| 5 | A2 / La2 | 110.00 Hz | 45 |
| 4 | D3 / Re3 | 146.83 Hz | 50 |
| 3 | G3 / Sol3 | 196.00 Hz | 55 |
| 2 | B3 / Si3 | 246.94 Hz | 59 |
| 1, la más aguda | E4 / Mi4 | 329.63 Hz | 64 |

MIDI es un estándar de representación musical. Aquí utilizaremos su numeración de
alturas: cada entero consecutivo aumenta un semitono. No se necesita un dispositivo
MIDI para utilizar estos números en Python.

Con esta afinación y sin técnicas que alteren la altura:

    número MIDI de la nota = número MIDI de la cuerda al aire + traste
    frecuencia = 440 × 2^((número MIDI - 69) / 12)

Los valores de frecuencia de la tabla proceden de esa fórmula. Se adopta La4=440 Hz
como referencia del proyecto. Una guitarra desafinada o un bend no cumple exactamente
este modelo, por lo que esas situaciones necesitan tratamiento adicional.

## 5. La misma nota puede tocarse en posiciones distintas

Cuerda 6, traste 5: 40 + 5 = 45, es decir, A2.
Cuerda 5, traste 0: 45 + 0 = 45, también A2.

Reconocer solamente 110 Hz no decide de forma única entre ambas posiciones. El
modelo intentará utilizar diferencias del sonido y contexto; eso no garantiza
recuperar siempre la digitación original. Evaluaremos altura y posición por separado
cuando sea necesario.

## 6. Sin nota y cuerda al aire son estados diferentes

- Traste 0: la cuerda está sonando al aire.
- Sin nota: no hay una nota anotada como activa en esa cuerda en ese instante.

«Sin nota» no significa que el micrófono mida cero absoluto: puede haber resonancias,
ruido y sonido de otras cuerdas. La correspondencia de esta clase con las anotaciones
se documentará en el cargador de datos.

Con trastes 0–19 tenemos 20 posiciones y, al añadir «sin nota», 21 clases por cuerda.
No usamos el traste 20 como posición válida dentro de este alcance. Si se elige 20
como código interno de «sin nota», se documentará que es un identificador de clase,
no un número de traste.

## 7. Una nota, varias notas y un acorde

Monofonía musical: una sola altura a la vez. Polifonía: varias alturas simultáneas.
La guitarra puede producir polifonía aunque la grabación tenga un solo canal de audio.
«Mono» como número de canales y «monofónico» como contenido musical no son equivalentes.

Un acorde es una combinación de notas con una interpretación armónica. Para esta
primera versión recuperaremos las notas y posiciones; asignar un nombre de acorde
requiere criterios adicionales y no se promete como salida inicial.

## 8. Lectura de una tablatura básica

Las seis líneas representan cuerdas. La superior es la cuerda 1, la más aguda, y la
inferior la cuerda 6. Los números indican trastes y se leen de izquierda a derecha.

Ejemplo didáctico de tres notas sucesivas, todas sobre la cuerda 6:

```text
e|----------------|
B|----------------|
G|----------------|
D|----------------|
A|----------------|
E|--0---1---3------|
```

Esto representa E2, F2 y G2. Los guiones proporcionan separación visual: por sí
solos no definen una duración musical exacta. Nuestro programa acompañará la tablatura
con una tabla de tiempos, y explicaremos su resolución temporal.

| Inicio ilustrativo | Final ilustrativo | Cuerda | Traste | Nota |
|---:|---:|---:|---:|---|
| 0.00 s | 0.40 s | 6 | 0 | E2 |
| 0.50 s | 0.90 s | 6 | 1 | F2 |
| 1.00 s | 1.40 s | 6 | 3 | G2 |

Los tres intervalos son ejemplos inventados con fines didácticos, no resultados.

## 9. Cómo se conecta con la red

La CQT convierte el audio en una tabla de tiempo y frecuencia. La CNN aprende
patrones de esa tabla. La capa recurrente relaciona observaciones consecutivas.
En cada paso temporal se producen seis distribuciones de 21 probabilidades, una
por cuerda. Cada distribución suma 1, no la suma global de todas las cuerdas.

Compartimos un extractor de características entre las seis tareas: las cabezas de
salida son distintas, pero no constituyen seis redes entrenadas sin relación.

Después se seleccionan estados, se suavizan con cuidado y se agrupan en eventos.
Una secuencia estable en el mismo traste no revela por sí sola si hubo una nueva
pulsación: se necesita detectar ataques o declarar esa limitación.

Para vectores internos propondremos orden grave→aguda, cuerdas [6, 5, 4, 3, 2, 1].
Al imprimir la tablatura se invierte a [1, 2, 3, 4, 5, 6]. El orden de las anotaciones
se verificará en los archivos antes de implementar esta correspondencia.

## 10. Comprobación de comprensión

1. ¿Por qué «traste 0» no significa silencio?
2. ¿Qué nota producen cuerda 6/traste 5 y cuerda 5/traste 0?
3. ¿Una grabación de un canal puede contener un acorde?
4. ¿Por qué un pico espectral puede ser un armónico y no otra nota?
5. ¿Por qué un estado repetido no prueba que hubo una sola pulsación?

Respuestas: 1) Es cuerda al aire sonando. 2) A2 en ambos casos. 3) Sí.
4) Una pulsación produce varios componentes relacionados. 5) Puede haber nuevos
ataques del mismo traste sin que cambie la clase predicha.

## Antecedente para ampliar

Wiggins, A., & Kim, Y. (2019). Guitar tablature estimation with a convolutional neural
network. En *Proceedings of the 20th International Society for Music Information
Retrieval Conference* (pp. 284–291). https://doi.org/10.5281/zenodo.3527800

Este artículo sustenta el antecedente de estimación por cuerda y la distinción entre
altura y digitación. Los ejemplos, cálculos y explicaciones de esta guía son didácticos.
