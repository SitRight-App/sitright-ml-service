# sitright-ml-service

Servicio de machine learning para clasificación postural del proyecto **SitRight**. FastAPI + Scikit-learn.

## Sobre el proyecto

Parte de la tesis **SitRight**: aplicación web con machine learning e IoT para mejorar la ergonomía postural en trabajadores sedentarios limeños mediante chaleco inteligente.

**Equipo:** Christopher Lecca, Mariano Ames (UPC — Ingeniería de Software).

## Qué hace

Expone un endpoint `POST /ml/classify` que recibe lecturas de 3 acelerómetros (cervical, dorsal, lumbar) y retorna la clase postural clasificada:

- `adequate` — postura correcta
- `forward_slouch` — encorvamiento frontal
- `excessive_recline` — reclinación excesiva hacia atrás
- `indeterminate` — confianza < 50%

## Modelo

Random Forest entrenado con Scikit-learn. Ver `ml-training/notebooks/` para el proceso completo.

Dataset base: `pkmandke/Human-Posture-Dataset` (IEEE IACC 2018). Enriquecido con datos del piloto en Lima.

## Relación con otros repos

- **sitright-backend-api** — lo llama vía HTTP POST cada 5 segundos (una vez por lectura del chaleco).
- **sitright-workspace** — documentación, ADRs y backlog (privado).

## Licencia

Proyecto académico — UPC 2026.
