# Stratic Synth

Полифонический wavetable-синтезатор (VST3 / Standalone) для Windows,
написанный с нуля на JUCE.

## Возможности

- **Осцилляторы**: 3 осциллятора — Sine / Saw / Square / Triangle / Wavetable / PWM / Noise
- **Wavetable**: встроенная таблица, импорт из `.wav`, редактор с рисованием волн,
  warp-режимы (Bend / Asym / Quantize / Smooth)
- **Sample-слой**: загрузка сэмплов, Root / Start / Reverse / Loop / OneShot, keytrack
- **Unison**: до 5 голосов, Detune, Spread
- **Фильтр**: LP / HP / BP / Notch + Filter Envelope + Drive + Keytrack
- **Модуляция**: LFO 1 + LFO 2 (фигуры, темп-синхронизация), 2 ADSR,
  Mod Matrix 4 слота × 12 источников (ModWheel, Aftertouch, MPE Timbre/Pressure и др.)
- **Голоса**: Poly / Mono / Legato + Glide, **MPE** (per-note bend / pressure / timbre)
- **Арпеджиатор**: 6 паттернов, темп-синк, октавы, gate
- **Эффекты**: Drive, Chorus, Delay, Reverb, 3-полосный EQ, Compressor, brickwall-лимитер
- **Пресеты**: 36 фабричных + пользовательский банк, save/load, экспорт/импорт банка
- **MIDI Learn** с сохранением маппингов в состоянии плагина
- **Мониторинг**: осциллограф, спектр-анализатор, стерео VU с peak-hold
- **UI**: тёмный масштабируемый интерфейс с анимацией и интерактивными дисплеями

## Сборка (Windows)

Требования:
- Windows 10/11
- Visual Studio 2022+ (workload «Desktop development with C++»)
- [JUCE](https://juce.com) 8.x

1. Клонируйте репозиторий:
git clone https://github.com/pilat102/StraticSynth.git
2. Откройте решение `Builds/VisualStudio2026/StraticSynth.sln`.
Если ваш JUCE лежит по другому пути — откройте `.juceproj` в Projucer
и пересохраните Visual Studio-экспорт.
3. Соберите конфигурацию `Release | x64`.
4. Результаты: VST3 — `Builds/VisualStudio2026/x64/Release/VST3`,
standalone — `...\Standalone Plugin\StraticSynth.exe`.

## Установка VST3

Скопируйте папку `StraticSynth.vst3` в каталог VST3 вашей DAW
или в системный `C:\Program Files\Common Files\VST3\`.

## Лицензия

Код распространяется под **GPL-3.0** (см. LICENSE).

> Проект использует JUCE (двойная лицензия AGPL / коммерческая).
> Для коммерческого распространения плагина необходима лицензия JUCE.