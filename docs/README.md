# Документация

## Исследования форматов

| Документ | Содержание | Связанный код |
|---|---|---|
| [Формат геометрии уровней DS2](ds2.md) | `DS2GeometryFile_1` версии `0.8`: секции, меши, буферы, материалы, лайтмапы, освещение и группы моделей | [`inspect_ds2.py`](../ds2_research/inspect_ds2.py), [`ds2_scene.py`](../ds2_research/ds2_scene.py), [`map_loader.cpp`](../viewer/map_loader.cpp) |
| [Форматы моделей DS2MD](ds2md.md) | Текстовый `0.5` и бинарные `0.6`, `0.9`, `1.0` | [`inspect_ds2md.py`](../ds2_research/inspect_ds2md.py), [`ds2md_legacy.py`](../ds2_research/ds2md_legacy.py), [`model_loader.cpp`](../viewer/model_loader.cpp) |
| [Октодерево vistree](vistree.md) | Структура узлов, связь с мешами, runtime-обход и отличие от PVS | [`inspect_ds2.py`](../ds2_research/inspect_ds2.py), [`map_loader.cpp`](../viewer/map_loader.cpp) |

## Дополнительные материалы

- [Шаблон DS2MD для ImHex](../formats/ds2md.hexpat).
- [Анимация DS2MD в Python](../ds2_research/ds2md_animation.py) — скелетные и sampled-matrix клипы без зависимости от рендерера.
- [Экспорт GLB](../viewer/glb_export.cpp) и [запись анимации GLB](../viewer/glb_animation.cpp).

## Как читать исследования

- Если не указано иное, многобайтовые значения little-endian, `f32` — IEEE 754, а `String` — `u16 byte_length + byte[byte_length]` без завершающего NUL.
- Адреса функций — VA при preferred image base `0x10000000`; RVA вычисляется как `VA - 0x10000000`.
- Пути `gameres/...` относятся к установленной игре. Сами ресурсы в репозиторий не входят.
- Утверждения о «корпусе» описывают только перечисленные исследованные файлы и не являются безусловными гарантиями для других версий игры.
- Исследовательские имена полей обозначают назначение, восстановленное по данным и потребителям; это не обязательно имена из исходного кода движка.

Практические команды сборки, запуска, тестирования и экспорта находятся в [корневом README](../README.md).
