#include "PluginEditor.h"

//==============================================================================
struct SimpleFFT
{
    static void process(std::vector<float>& re, std::vector<float>& im)
    {
        const size_t n = re.size();

        for (size_t i = 1, j = 0; i < n; ++i)
        {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;

            if (i < j)
            {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }

        for (size_t len = 2; len <= n; len <<= 1)
        {
            const float ang = -2.0f * juce::MathConstants<float>::pi / static_cast<float>(len);
            const float wr = std::cos(ang);
            const float wi = std::sin(ang);

            for (size_t i = 0; i < n; i += len)
            {
                float cwr = 1.0f, cwi = 0.0f;

                for (size_t k = 0; k < len / 2; ++k)
                {
                    const size_t a = i + k;
                    const size_t b = i + k + len / 2;

                    const float tr = re[b] * cwr - im[b] * cwi;
                    const float ti = re[b] * cwi + im[b] * cwr;

                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;

                    const float nwr = cwr * wr - cwi * wi;
                    cwi = cwr * wi + cwi * wr;
                    cwr = nwr;
                }
            }
        }
    }
};

//==============================================================================
void OutputMeter::computeSpectrum()
{
    for (int i = 0; i < 1024; ++i)
    {
        fftRe[static_cast<size_t>(i)] = bufL[static_cast<size_t>(i)] * win[static_cast<size_t>(i)];
        fftIm[static_cast<size_t>(i)] = 0.0f;
    }

    SimpleFFT::process(fftRe, fftIm);

    const int numBars = static_cast<int>(bars.size());
    const float fMin = 40.0f;
    const float fMax = 16000.0f;
    const float nyq = sr * 0.5f;

    for (int b = 0; b < numBars; ++b)
    {
        const float f0 = fMin * std::pow(fMax / fMin, static_cast<float>(b) / numBars);
        const float f1 = fMin * std::pow(fMax / fMin, static_cast<float>(b + 1) / numBars);

        int i0 = static_cast<int>(f0 * 1024.0f / nyq);
        int i1 = static_cast<int>(f1 * 1024.0f / nyq);

        i0 = juce::jlimit(0, 511, i0);
        i1 = juce::jlimit(i0 + 1, 512, i1);

        float peak = 0.0f;

        for (int i = i0; i < i1; ++i)
        {
            const float m = std::sqrt(fftRe[static_cast<size_t>(i)] * fftRe[static_cast<size_t>(i)] +
                fftIm[static_cast<size_t>(i)] * fftIm[static_cast<size_t>(i)]);
            peak = juce::jmax(peak, m / 512.0f);
        }

        const float db = peak > 1e-6f ? juce::Decibels::gainToDecibels(peak) : -100.0f;
        const float v = juce::jlimit(0.0f, 1.0f, (db + 70.0f) / 70.0f);

        bars[static_cast<size_t>(b)] = juce::jmax(v, bars[static_cast<size_t>(b)] * 0.80f);
    }
}

//==============================================================================
void StraticLookAndFeel::drawRotarySlider(juce::Graphics& g,
    int x, int y, int width, int height,
    float sliderPosProportional,
    float rotaryStartAngle,
    float rotaryEndAngle,
    juce::Slider& slider)
{
    const float diameter = static_cast<float>(juce::jmin(width, height));
    const float cx = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float radius = diameter * 0.5f - 4.0f;

    const bool active = slider.isMouseOver() || slider.isMouseButtonDown();

    g.setColour(juce::Colour(0x55000000));
    g.fillEllipse(cx - radius, cy - radius + 2.0f, radius * 2.0f, radius * 2.0f);

    const int numTicks = 11;
    for (int i = 0; i < numTicks; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(numTicks - 1);
        const float a = rotaryStartAngle + t * (rotaryEndAngle - rotaryStartAngle);

        const float r1 = radius + 1.0f;
        const float r2 = radius + 3.5f;

        juce::Line<float> tick(std::cos(a) * r1 + cx, std::sin(a) * r1 + cy,
            std::cos(a) * r2 + cx, std::sin(a) * r2 + cy);

        g.setColour(t <= sliderPosProportional ? juce::Colour(0x8800e5ff)
            : juce::Colour(0x2affffff));
        g.drawLine(tick, 1.0f);
    }

    juce::Path track;
    track.addCentredArc(cx, cy, radius - 3.0f, radius - 3.0f, 0.0f,
        rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(juce::Colour(0xff222834));
    g.strokePath(track, juce::PathStrokeType(4.5f,
        juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    const float toAngle = rotaryStartAngle +
        sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    juce::Path value;
    value.addCentredArc(cx, cy, radius - 3.0f, radius - 3.0f, 0.0f,
        rotaryStartAngle, toAngle, true);

    g.setColour(active ? juce::Colour(0x6633f0ff) : juce::Colour(0x4400e5ff));
    g.strokePath(value, juce::PathStrokeType(8.0f,
        juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    g.setColour(active ? juce::Colour(0xff33f0ff) : juce::Colour(0xff00e5ff));
    g.strokePath(value, juce::PathStrokeType(3.0f,
        juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    const auto end = value.getCurrentPosition();
    g.setColour(juce::Colour(0x6600e5ff));
    g.fillEllipse(end.x - 4.0f, end.y - 4.0f, 8.0f, 8.0f);
    g.setColour(juce::Colours::white);
    g.fillEllipse(end.x - 2.0f, end.y - 2.0f, 4.0f, 4.0f);

    juce::ColourGradient cap(juce::Colour(0xff2b3240), cx, cy - radius,
        juce::Colour(0xff161b24), cx, cy + radius, false);
    g.setGradientFill(cap);
    g.fillEllipse(cx - radius * 0.62f, cy - radius * 0.62f,
        radius * 1.24f, radius * 1.24f);

    juce::Path highlight;
    highlight.addCentredArc(cx, cy, radius * 0.52f, radius * 0.52f, 0.0f, -2.6f, -0.5f, true);
    g.setColour(juce::Colour(0x22ffffff));
    g.strokePath(highlight, juce::PathStrokeType(1.2f));

    juce::Line<float> pointer(cx + std::cos(toAngle) * radius * 0.18f,
        cy + std::sin(toAngle) * radius * 0.18f,
        cx + std::cos(toAngle) * radius * 0.52f,
        cy + std::sin(toAngle) * radius * 0.52f);
    g.setColour(active ? juce::Colour(0xff33f0ff) : juce::Colour(0xff9fb2ff));
    g.drawLine(pointer, 2.0f);

    g.setColour(juce::Colour(0xffdfe6f2));
    g.setFont(juce::Font(juce::FontOptions(9.0f)));
    g.drawText(slider.getTextFromValue(slider.getValue()),
        juce::Rectangle<float>(cx - radius, cy + radius * 0.12f,
            radius * 2.0f, radius * 0.8f).toNearestInt(),
        juce::Justification::centred);
}

//==============================================================================
void StraticLookAndFeel::drawComboBox(juce::Graphics& g,
    int width, int height,
    bool, int, int, int, int,
    juce::ComboBox& box)
{
    const auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat();

    g.setColour(juce::Colour(0xff1a202b));
    g.fillRoundedRectangle(bounds, 6.0f);

    const bool active = box.isMouseOver() || box.isMouseButtonDown();
    g.setColour(active ? juce::Colour(0xff00e5ff) : juce::Colour(0xff39424f));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.5f);

    juce::Path arrow;
    const float ax = static_cast<float>(width) - 20.0f;
    const float ay = static_cast<float>(height) * 0.5f;
    arrow.addTriangle(ax, ay - 3.0f, ax + 10.0f, ay - 3.0f, ax + 5.0f, ay + 4.0f);
    g.setColour(juce::Colour(0xff00e5ff));
    g.fillPath(arrow);
}

//==============================================================================
void StraticLookAndFeel::drawButtonBackground(juce::Graphics& g,
    juce::Button& button,
    const juce::Colour&,
    bool highlighted,
    bool down)
{
    const auto bounds = button.getLocalBounds().toFloat();

    const juce::Colour top = down ? juce::Colour(0xff10151d)
        : highlighted ? juce::Colour(0xff232b39)
        : juce::Colour(0xff1a202b);
    const juce::Colour bottom = down ? juce::Colour(0xff0b0f15)
        : highlighted ? juce::Colour(0xff1b2230)
        : juce::Colour(0xff141922);

    juce::ColourGradient grad(top, bounds.getX(), bounds.getY(),
        bottom, bounds.getX(), bounds.getBottom(), false);
    g.setGradientFill(grad);
    g.fillRoundedRectangle(bounds, 6.0f);

    g.setColour(highlighted ? juce::Colour(0xff00e5ff) : juce::Colour(0xff39424f));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.0f);

    if (highlighted)
    {
        g.setColour(juce::Colour(0x2200e5ff));
        g.drawRoundedRectangle(bounds.reduced(-1.5f), 7.0f, 2.0f);
    }
}

//==============================================================================
WavetableDisplay::WavetableDisplay(juce::AudioProcessorValueTreeState& state,
    std::atomic<WavetableData*>* source)
{
    wtAtomic = state.getRawParameterValue("wtPos");
    wtParam = dynamic_cast<juce::RangedAudioParameter*>(state.getParameter("wtPos"));
    wtSource = source;
    startTimerHz(30);
}

const WavetableData& WavetableDisplay::activeTable() const
{
    WavetableData* imp = wtSource != nullptr ? wtSource->load() : nullptr;
    if (imp != nullptr && !imp->frames.empty())
        return *imp;
    return builtIn;
}

void WavetableDisplay::timerCallback()
{
    const float v = wtAtomic != nullptr ? wtAtomic->load() : 0.5f;
    void* p = wtSource != nullptr ? static_cast<void*>(wtSource->load()) : nullptr;

    if (std::abs(v - lastValue) > 0.0001f || p != lastPtr || stackView != lastStack)
    {
        lastValue = v;
        lastPtr = p;
        lastStack = stackView;
        repaint();
    }
}

void WavetableDisplay::mouseDown(const juce::MouseEvent& e)
{
    if (wtParam != nullptr) wtParam->beginChangeGesture();
    dragTo(e);
}

void WavetableDisplay::mouseDrag(const juce::MouseEvent& e) { dragTo(e); }

void WavetableDisplay::mouseUp(const juce::MouseEvent&)
{
    if (wtParam != nullptr) wtParam->endChangeGesture();
}

void WavetableDisplay::mouseDoubleClick(const juce::MouseEvent&)
{
    stackView = !stackView;
    repaint();
}

void WavetableDisplay::mouseWheelMove(const juce::MouseEvent&,
    const juce::MouseWheelDetails& wheel)
{
    if (wtParam == nullptr) return;

    const int nf = activeTable().numFrames;
    const float step = nf > 1 ? 1.0f / static_cast<float>(nf - 1) : 0.1f;

    const float cur = wtAtomic != nullptr ? wtAtomic->load() : 0.5f;
    wtParam->setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, cur + wheel.deltaY * step));
}

void WavetableDisplay::dragTo(const juce::MouseEvent& e)
{
    const float w = static_cast<float>(getWidth());
    if (w < 8.0f || wtParam == nullptr) return;

    wtParam->setValueNotifyingHost(juce::jlimit(
        0.0f, 1.0f, (static_cast<float>(e.getPosition().x) - 6.0f) / (w - 12.0f)));
}

void WavetableDisplay::paint(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();

    juce::ColourGradient grad(juce::Colour(0xff141922), 0.0f, 0.0f,
        juce::Colour(0xff0c0f16), 0.0f, bounds.getHeight(), false);
    g.setGradientFill(grad);
    g.fillAll();

    const float wt = wtAtomic != nullptr ? wtAtomic->load() : 0.5f;
    const WavetableData& table = activeTable();

    WavetableData* impPtr = wtSource != nullptr ? wtSource->load() : nullptr;
    const bool imported = impPtr != nullptr && !impPtr->frames.empty();

    g.setColour(juce::Colour(0xff9fb2ff));
    g.setFont(juce::Font(juce::FontOptions(12.0f)));
    g.drawText("WAVETABLE  " + juce::String(imported ? "imported" : "internal")
        + "  " + juce::String(table.numFrames) + " fr",
        10, 6, 320, 16, juce::Justification::left);

    g.setColour(juce::Colour(0xff8b93a7));
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    g.drawText(stackView ? "3D  (dbl-click: 2D)" : "2D  (dbl-click: 3D)",
        340, 8, 150, 14, juce::Justification::left);

    g.setColour(juce::Colour(0xff00e5ff));
    g.setFont(juce::Font(juce::FontOptions(12.0f)));
    g.drawText(juce::String(static_cast<int>(wt * 100.0f)) + " %",
        getLocalBounds().removeFromRight(80).withY(6).withHeight(16),
        juce::Justification::centredRight);

    auto waveArea = bounds.withTrimmedTop(24.0f).withTrimmedBottom(42.0f).reduced(6.0f, 0.0f);

    g.setColour(juce::Colour(0x14ffffff));
    for (int i = 1; i < 8; ++i)
    {
        const float gx = waveArea.getX() + waveArea.getWidth() * (static_cast<float>(i) / 8.0f);
        g.drawVerticalLine(static_cast<int>(gx), waveArea.getY(), waveArea.getBottom());
    }

    const int n = static_cast<int>(waveArea.getWidth()) + 1;

    if (n > 2 && table.tableSize > 1)
    {
        if (stackView)
        {
            const int F = table.numFrames;
            const int step = juce::jmax(1, F / 14);

            for (int f = F - 1; f >= 0; f -= step)
            {
                const float t = F > 1 ? static_cast<float>(f) / static_cast<float>(F - 1) : 0.0f;

                const float x0 = waveArea.getX() + t * waveArea.getWidth() * 0.18f;
                const float drawW = waveArea.getWidth() * 0.82f;
                const float yBase = waveArea.getBottom() - t * waveArea.getHeight() * 0.55f;
                const float amp = waveArea.getHeight() * 0.16f;

                const bool active = std::abs(t - wt) <= (0.5f * step / juce::jmax(1, F - 1));

                juce::Path path;
                for (int i = 0; i < n; ++i)
                {
                    const float phase = static_cast<float>(i) / static_cast<float>(n - 1);
                    const float v = table.getSample(phase, t);
                    const float px = x0 + drawW * phase;
                    const float py = yBase - v * amp;
                    if (i == 0) path.startNewSubPath(px, py);
                    else path.lineTo(px, py);
                }

                if (active) { g.setColour(juce::Colour(0xff00e5ff)); g.strokePath(path, juce::PathStrokeType(2.0f)); }
                else { g.setColour(juce::Colour(0x557f8ba3)); g.strokePath(path, juce::PathStrokeType(1.0f)); }
            }
        }
        else
        {
            juce::Path path;
            for (int i = 0; i < n; ++i)
            {
                const float phase = static_cast<float>(i) / static_cast<float>(n - 1);
                const float v = table.getSample(phase, wt);
                const float px = waveArea.getX() + static_cast<float>(i);
                const float py = waveArea.getCentreY() - v * waveArea.getHeight() * 0.45f;
                if (i == 0) path.startNewSubPath(px, py);
                else path.lineTo(px, py);
            }

            g.setColour(juce::Colour(0x3300e5ff));
            g.strokePath(path, juce::PathStrokeType(6.0f));
            g.setColour(juce::Colour(0xff00e5ff));
            g.strokePath(path, juce::PathStrokeType(2.2f));
        }
    }

    auto strip = bounds.withTrimmedTop(bounds.getHeight() - 36.0f).reduced(6.0f, 4.0f);
    const int cells = juce::jlimit(1, 32, table.numFrames);
    const float cellW = strip.getWidth() / static_cast<float>(cells);

    for (int f = 0; f < cells; ++f)
    {
        const float framePos = cells > 1 ? static_cast<float>(f) / static_cast<float>(cells - 1) : 0.0f;

        auto cell = juce::Rectangle<float>(strip.getX() + cellW * static_cast<float>(f),
            strip.getY(), cellW, strip.getHeight());

        const bool active = std::abs(wt - framePos) <= (0.5f / juce::jmax(1, cells - 1));

        juce::Path miniPath;
        const int miniN = 48;
        for (int i = 0; i < miniN; ++i)
        {
            const float phase = static_cast<float>(i) / static_cast<float>(miniN - 1);
            const float v = table.getSample(phase, framePos);
            const float px = cell.getX() + 2.0f + (cell.getWidth() - 4.0f) * phase;
            const float py = cell.getCentreY() - v * cell.getHeight() * 0.40f;
            if (i == 0) miniPath.startNewSubPath(px, py);
            else miniPath.lineTo(px, py);
        }

        g.setColour(active ? juce::Colour(0xff00e5ff) : juce::Colour(0x667f8ba3));
        g.strokePath(miniPath, juce::PathStrokeType(1.2f));

        if (active)
        {
            g.setColour(juce::Colour(0x5500e5ff));
            g.drawRoundedRectangle(cell.reduced(1.0f), 3.0f, 1.0f);
        }
    }

    g.setColour(juce::Colour(0xff2a3140));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 8.0f, 1.0f);
}

//==============================================================================
EnvelopeDisplay::EnvelopeDisplay(juce::AudioProcessorValueTreeState& state)
{
    aAtomic = state.getRawParameterValue("attack");
    dAtomic = state.getRawParameterValue("decay");
    sAtomic = state.getRawParameterValue("sustain");
    rAtomic = state.getRawParameterValue("release");

    aParam = dynamic_cast<juce::RangedAudioParameter*>(state.getParameter("attack"));
    dParam = dynamic_cast<juce::RangedAudioParameter*>(state.getParameter("decay"));
    sParam = dynamic_cast<juce::RangedAudioParameter*>(state.getParameter("sustain"));
    rParam = dynamic_cast<juce::RangedAudioParameter*>(state.getParameter("release"));

    startTimerHz(30);
}

void EnvelopeDisplay::setParam(juce::RangedAudioParameter* p, float value)
{
    if (p != nullptr)
        p->setValueNotifyingHost(p->convertTo0to1(value));
}

EnvelopeDisplay::EnvLayout EnvelopeDisplay::computeLayout() const
{
    EnvLayout L;

    const float at = aAtomic != nullptr ? aAtomic->load() : 0.01f;
    const float dc = dAtomic != nullptr ? dAtomic->load() : 0.2f;
    const float sus = sAtomic != nullptr ? sAtomic->load() : 0.7f;
    const float re = rAtomic != nullptr ? rAtomic->load() : 0.3f;

    const float W = static_cast<float>(getWidth()) - 16.0f;

    L.maxA = 0.30f * W;
    L.maxD = 0.30f * W;
    L.maxR = 0.22f * W;

    L.yTop = 26.0f;
    L.yBot = static_cast<float>(getHeight()) - 12.0f;
    L.yS = L.yTop + (1.0f - sus) * (L.yBot - L.yTop);

    L.x0 = 8.0f;
    L.x1 = L.x0 + juce::jmax(4.0f, L.maxA * std::sqrt(at / 5.0f));
    L.x2 = L.x1 + juce::jmax(4.0f, L.maxD * std::sqrt(dc / 5.0f));
    L.x3 = L.x2 + 0.18f * W;
    L.x4 = L.x3 + juce::jmax(4.0f, L.maxR * std::sqrt(re / 8.0f));

    return L;
}

void EnvelopeDisplay::timerCallback()
{
    float snapshot[4];
    snapshot[0] = aAtomic != nullptr ? aAtomic->load() : 0.0f;
    snapshot[1] = dAtomic != nullptr ? dAtomic->load() : 0.0f;
    snapshot[2] = sAtomic != nullptr ? sAtomic->load() : 0.0f;
    snapshot[3] = rAtomic != nullptr ? rAtomic->load() : 0.0f;

    bool changed = false;
    for (int i = 0; i < 4; ++i)
        if (std::abs(snapshot[i] - lastSnapshot[i]) > 0.0001f)
            changed = true;

    if (changed)
    {
        for (int i = 0; i < 4; ++i) lastSnapshot[i] = snapshot[i];
        repaint();
    }
}

void EnvelopeDisplay::mouseDown(const juce::MouseEvent& e)
{
    const auto L = computeLayout();
    const auto p = e.getPosition().toFloat();

    const juce::Point<float> handles[3] = {
        { L.x1, L.yTop }, { L.x2, L.yS }, { L.x3, L.yS } };

    activeHandle = -1;
    for (int i = 0; i < 3; ++i)
        if (p.getDistanceFrom(handles[i]) < 14.0f) { activeHandle = i; break; }

    if (activeHandle == 0 && aParam) aParam->beginChangeGesture();
    if (activeHandle == 1) { if (dParam) dParam->beginChangeGesture(); if (sParam) sParam->beginChangeGesture(); }
    if (activeHandle == 2 && rParam) rParam->beginChangeGesture();
}

void EnvelopeDisplay::mouseDrag(const juce::MouseEvent& e)
{
    if (activeHandle < 0) return;

    const auto L = computeLayout();
    const auto p = e.getPosition().toFloat();

    if (activeHandle == 0)
    {
        const float wA = juce::jlimit(4.0f, L.maxA, p.x - L.x0);
        const float t = wA / L.maxA;
        setParam(aParam, 5.0f * t * t);
    }
    else if (activeHandle == 1)
    {
        const float wD = juce::jlimit(4.0f, L.maxD, p.x - L.x1);
        const float t = wD / L.maxD;
        setParam(dParam, 5.0f * t * t);
        setParam(sParam, juce::jlimit(0.0f, 1.0f,
            1.0f - (p.y - L.yTop) / juce::jmax(1.0f, L.yBot - L.yTop)));
    }
    else if (activeHandle == 2)
    {
        const float wR = juce::jlimit(4.0f, L.maxR, p.x - L.x3);
        const float t = wR / L.maxR;
        setParam(rParam, 8.0f * t * t);
    }
}

void EnvelopeDisplay::mouseUp(const juce::MouseEvent&)
{
    if (activeHandle == 0 && aParam) aParam->endChangeGesture();
    if (activeHandle == 1) { if (dParam) dParam->endChangeGesture(); if (sParam) sParam->endChangeGesture(); }
    if (activeHandle == 2 && rParam) rParam->endChangeGesture();
    activeHandle = -1;
}

void EnvelopeDisplay::paint(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();

    juce::ColourGradient grad(juce::Colour(0xff141922), 0.0f, 0.0f,
        juce::Colour(0xff0c0f16), 0.0f, bounds.getHeight(), false);
    g.setGradientFill(grad);
    g.fillAll();

    const auto L = computeLayout();

    g.setColour(juce::Colour(0xff9fb2ff));
    g.setFont(juce::Font(juce::FontOptions(12.0f)));
    g.drawText("ENVELOPE", 10, 6, 200, 16, juce::Justification::left);

    const float at = aAtomic != nullptr ? aAtomic->load() : 0.01f;
    const float dc = dAtomic != nullptr ? dAtomic->load() : 0.2f;
    const float sus = sAtomic != nullptr ? sAtomic->load() : 0.7f;
    const float re = rAtomic != nullptr ? rAtomic->load() : 0.3f;

    g.setColour(juce::Colour(0xff8b93a7));
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    g.drawText("A " + juce::String(at, 2) + "  D " + juce::String(dc, 2) +
        "  S " + juce::String(sus, 2) + "  R " + juce::String(re, 2),
        getLocalBounds().removeFromRight(190).withY(6).withHeight(16),
        juce::Justification::centredRight);

    juce::Path fill;
    fill.startNewSubPath(L.x0, L.yBot);
    fill.lineTo(L.x1, L.yTop);
    fill.lineTo(L.x2, L.yS);
    fill.lineTo(L.x3, L.yS);
    fill.lineTo(L.x4, L.yBot);
    fill.closeSubPath();

    juce::ColourGradient fillGrad(juce::Colour(0x5500e5ff), 0.0f, L.yTop,
        juce::Colour(0x0500e5ff), 0.0f, L.yBot, false);
    g.setGradientFill(fillGrad);
    g.fillPath(fill);

    juce::Path line;
    line.startNewSubPath(L.x0, L.yBot);
    line.lineTo(L.x1, L.yTop);
    line.lineTo(L.x2, L.yS);
    line.lineTo(L.x3, L.yS);
    line.lineTo(L.x4, L.yBot);
    g.setColour(juce::Colour(0xff00e5ff));
    g.strokePath(line, juce::PathStrokeType(2.0f));

    const juce::Point<float> handles[3] = {
        { L.x1, L.yTop }, { L.x2, L.yS }, { L.x3, L.yS } };

    for (int i = 0; i < 3; ++i)
    {
        g.setColour(juce::Colour(0x6600e5ff));
        g.fillEllipse(handles[i].x - 7.0f, handles[i].y - 7.0f, 14.0f, 14.0f);
        g.setColour(juce::Colours::white);
        g.fillEllipse(handles[i].x - 3.5f, handles[i].y - 3.5f, 7.0f, 7.0f);
    }

    g.setColour(juce::Colour(0xff2a3140));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 8.0f, 1.0f);
}

//==============================================================================
juce::AudioProcessorEditor* StraticSynthAudioProcessor::createEditor()
{
    return new StraticSynthAudioProcessorEditor(*this);
}

//==============================================================================
StraticSynthAudioProcessorEditor::StraticSynthAudioProcessorEditor(StraticSynthAudioProcessor& processor)
    : juce::AudioProcessorEditor(&processor),
    synthProcessor(processor),
    wavetableDisplay(processor.parameters, processor.getWavetableSource()),
    envelopeDisplay(processor.parameters),
    outputMeter(&processor.getMonitor()),
    wtEditor(processor)
{
    setLookAndFeel(&look);
    addAndMakeVisible(content);

    bgPanel.setBounds(0, 0, 1000, 1050);
    content.addAndMakeVisible(bgPanel);
    bgPanel.setSources(processor.getModWheelSource(), processor.getAftertouchSource());

    look.setColour(juce::ResizableWindow::backgroundColourId, juce::Colour(0xff0b0e14));
    look.setColour(juce::Label::textColourId, juce::Colours::white);
    look.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff1a202b));
    look.setColour(juce::ComboBox::textColourId, juce::Colours::white);
    look.setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff39424f));
    look.setColour(juce::ComboBox::arrowColourId, juce::Colour(0xff00e5ff));
    look.setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xff181c26));
    look.setColour(juce::PopupMenu::textColourId, juce::Colours::white);
    look.setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(0xff00bcd4));
    look.setColour(juce::PopupMenu::highlightedTextColourId, juce::Colours::black);
    look.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff1a202b));
    look.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff232a38));
    look.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    look.setColour(juce::TextButton::textColourOnId, juce::Colour(0xff00e5ff));

    titleLabel.setText("STRATIC SYNTH", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(juce::FontOptions(20.0f)));
    titleLabel.setColour(juce::Label::textColourId, juce::Colour(0xff00e5ff));
    content.addAndMakeVisible(titleLabel);

    presetNameLabel.setFont(juce::Font(juce::FontOptions(14.0f)));
    presetNameLabel.setColour(juce::Label::textColourId, juce::Colour(0xff9fb2ff));
    content.addAndMakeVisible(presetNameLabel);

    content.addAndMakeVisible(wavetableDisplay);
    content.addAndMakeVisible(envelopeDisplay);
    content.addAndMakeVisible(outputMeter);

    presetBox.setEditableText(false);
    for (int i = 0; i < synthProcessor.getNumPresets(); ++i)
        presetBox.addItem(synthProcessor.getPresetName(i), i + 1);
    presetBox.onChange = [this] { synthProcessor.loadPresetByIndex(presetBox.getSelectedId() - 1); };
    content.addAndMakeVisible(presetBox);

    prevButton.onClick = [this] { synthProcessor.prevPreset(); };
    nextButton.onClick = [this] { synthProcessor.nextPreset(); };

    saveButton.onClick = [this]
        {
            const auto current = synthProcessor.getCurrentPresetName();

            auto* box = new juce::AlertWindow("Save user preset",
                "Preset name:",
                juce::MessageBoxIconType::QuestionIcon);

            box->addTextEditor("name", current);
            box->addButton("Save", 1);
            box->addButton("Cancel", 0);

            box->enterModalState(true,
                juce::ModalCallbackFunction::create([this, box](int result)
                    {
                        if (result == 1)
                        {
                            const auto name = box->getTextEditorContents("name").trim();

                            if (name.isNotEmpty())
                                synthProcessor.saveCurrentAsUserPreset(name);
                        }
                    }));
        };

    loadButton.onClick = [this]
        {
            auto chooser = std::make_shared<juce::FileChooser>(
                "Load Stratic preset",
                juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                "*.straticp");

            chooser->launchAsync(
                juce::FileBrowserComponent::openMode |
                juce::FileBrowserComponent::canSelectFiles,
                [this, chooser](const juce::FileChooser& fc)
                {
                    auto file = fc.getResult();
                    if (file != juce::File())
                        synthProcessor.loadPresetFromFile(file);
                });
        };

    for (auto* b : { &prevButton, &nextButton, &saveButton, &loadButton })
        content.addAndMakeVisible(b);

    //==========================================================================
    auto setupKnob = [&](juce::Slider& slider, juce::Label& label,
        const juce::String& text, const juce::String& paramID)
        {
            slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
            slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
            content.addAndMakeVisible(slider);

            label.setText(text, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centred);
            label.setFont(juce::Font(juce::FontOptions(11.0f)));
            label.setColour(juce::Label::textColourId, juce::Colour(0xffc7cfdd));
            content.addAndMakeVisible(label);

            sliderAttachments.push_back(
                std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                    synthProcessor.parameters, paramID, slider));

            learnTargets[&slider] = paramID;
            paramLabelMap[paramID] = &label;
        };

    auto setupCombo = [&](juce::ComboBox& combo, const juce::String& paramID,
        const juce::StringArray& items)
        {
            combo.setEditableText(false);
            combo.clear(juce::dontSendNotification);
            for (int i = 0; i < items.size(); ++i)
                combo.addItem(items[i], i + 1);
            content.addAndMakeVisible(combo);

            comboAttachments.push_back(
                std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
                    synthProcessor.parameters, paramID, combo));
        };

    auto setupTitle = [&](juce::Label& label, const juce::String& text)
        {
            label.setText(text, juce::dontSendNotification);
            label.setFont(juce::Font(juce::FontOptions(13.0f)));
            label.setColour(juce::Label::textColourId, juce::Colour(0xff9fb2ff));
            content.addAndMakeVisible(label);
        };

    auto setupSmallLabel = [&](juce::Label& label, const juce::String& text)
        {
            label.setText(text, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centred);
            label.setFont(juce::Font(juce::FontOptions(10.0f)));
            label.setColour(juce::Label::textColourId, juce::Colour(0xffc7cfdd));
            content.addAndMakeVisible(label);
        };

    //==========================================================================
    const juce::String shapes[7] = { "Sine", "Saw", "Square", "Triangle",
                                     "Wavetable", "PWM", "Noise" };

    for (int i = 0; i < 3; ++i)
    {
        setupTitle(oscTitles[i], "OSC " + juce::String(i + 1));
        setupCombo(oscBoxes[i], "osc" + juce::String(i + 1) + "Shape",
            juce::StringArray(shapes, 7));
        setupKnob(levelSliders[i], levelLabels[i], "Level",
            "osc" + juce::String(i + 1) + "Level");
    }

    setupKnob(auxSliders[0], auxLabels[0], "Detune", "detune");
    setupKnob(auxSliders[1], auxLabels[1], "Semi", "osc3Semi");

    setupTitle(filterTitle, "LFO 1");
    setupKnob(lfoRateSlider, lfoRateLabel, "Rate", "lfoRate");
    setupKnob(lfoDepthSlider, lfoDepthLabel, "Amount", "lfoDepth");

    setupTitle(masterTitle, "WAVETABLE / MASTER");
    setupKnob(wtPosSlider, wtPosLabel, "WT Pos", "wtPos");
    setupKnob(volumeSlider, volumeLabel, "Volume", "volume");
    setupKnob(glideSlider, glideLabel, "Glide", "glide");
    setupCombo(voiceModeBox, "voiceMode", { "Poly", "Mono", "Legato" });

    setupTitle(uniTitle, "UNISON");
    setupKnob(uniVoicesSlider, uniVoicesLabel, "Voices", "uniVoices");
    setupKnob(uniDetuneSlider, uniDetuneLabel, "Detune", "uniDetune");
    setupKnob(uniSpreadSlider, uniSpreadLabel, "Spread", "uniSpread");

    setupTitle(sampleTitle, "SAMPLE (no file)");
    setupKnob(sampleLevelSlider, sampleLevelLabel, "Level", "sampleLevel");
    setupKnob(sampleRootSlider, sampleRootLabel, "Root", "sampleRoot");
    setupKnob(sampleStartSlider, sampleStartLabel, "Start", "sampleStart");
    setupCombo(sampleModeBox, "sampleMode", { "OneShot", "Loop" });
    setupCombo(sampleRevBox, "sampleReverse", { "Off", "On" });
    setupSmallLabel(sampleModeLabel, "Mode");
    setupSmallLabel(sampleRevLabel, "Rev");

    loadSampleButton.onClick = [this]
        {
            auto chooser = std::make_shared<juce::FileChooser>(
                "Load sample",
                juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                "*.wav;*.aiff;*.aif;*.mp3;*.flac;*.ogg");

            chooser->launchAsync(
                juce::FileBrowserComponent::openMode |
                juce::FileBrowserComponent::canSelectFiles,
                [this, chooser](const juce::FileChooser& fc)
                {
                    auto file = fc.getResult();
                    if (file != juce::File())
                        synthProcessor.loadSampleFromFile(file);
                });
        };
    content.addAndMakeVisible(loadSampleButton);

    loadWtButton.onClick = [this]
        {
            auto chooser = std::make_shared<juce::FileChooser>(
                "Load wavetable",
                juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                "*.wav;*.aiff;*.aif");

            chooser->launchAsync(
                juce::FileBrowserComponent::openMode |
                juce::FileBrowserComponent::canSelectFiles,
                [this, chooser](const juce::FileChooser& fc)
                {
                    auto file = fc.getResult();
                    if (file != juce::File())
                        synthProcessor.loadWavetableFromFile(file);
                });
        };
    content.addAndMakeVisible(loadWtButton);

    resetWtButton.onClick = [this] { synthProcessor.resetWavetable(); };
    content.addAndMakeVisible(resetWtButton);

    editWtButton.onClick = [this]
        {
            wtEditor.open(synthProcessor.getWavetableSource()->load());
        };
    content.addAndMakeVisible(editWtButton);

        bankButton.onClick = [this]
    {
        juce::PopupMenu menu;
        menu.addItem(1, "Export bank...");
        menu.addItem(2, "Import bank...");

        menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&bankButton),
                           [this](int result)
                           {
                               if (result == 1)
                               {
                                   auto chooser = std::make_shared<juce::FileChooser>(
                                       "Export Stratic bank",
                                       juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                                       "*.straticbank");

                                   chooser->launchAsync(
                                       juce::FileBrowserComponent::saveMode |
                                       juce::FileBrowserComponent::warnAboutOverwriting,
                                       [this, chooser](const juce::FileChooser& fc)
                                       {
                                           auto file = fc.getResult();
                                           if (file != juce::File())
                                               synthProcessor.exportBankToFile(file);
                                       });
                               }
                               else if (result == 2)
                               {
                                   auto chooser = std::make_shared<juce::FileChooser>(
                                       "Import Stratic bank",
                                       juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                                       "*.straticbank");

                                   chooser->launchAsync(
                                       juce::FileBrowserComponent::openMode |
                                       juce::FileBrowserComponent::canSelectFiles,
                                       [this, chooser](const juce::FileChooser& fc)
                                       {
                                           auto file = fc.getResult();
                                           if (file != juce::File())
                                               synthProcessor.importBankFromFile(file);
                                       });
                               }
                           });
    };
    content.addAndMakeVisible(bankButton);

    //==========================================================================
    setupTitle(filterPanelTitle, "FILTER");
    setupCombo(filterTypeBox, "filterType", { "LP", "HP", "BP", "Notch" });
    setupKnob(cutoffSlider, cutoffLabel, "Cutoff", "cutoff");
    setupKnob(resonanceSlider, resonanceLabel, "Res", "resonance");
    setupKnob(filterDriveSlider, filterDriveLabel, "Drive", "filterDrive");
    setupKnob(keytrackSlider, keytrackLabel, "Keytrk", "filterKeytrack");
    setupKnob(fenvAmtSlider, fenvAmtLabel, "EnvAmt", "fenvAmount");
    setupKnob(fenvASlider, fenvALabel, "F.Atk", "fenvAttack");
    setupKnob(fenvDSlider, fenvDLabel, "F.Dec", "fenvDecay");
    setupKnob(fenvSSlider, fenvSLabel, "F.Sus", "fenvSustain");
    setupKnob(fenvRSlider, fenvRLabel, "F.Rel", "fenvRelease");

    setupTitle(eqTitle, "EQ");
    setupKnob(eqLowSlider, eqLowLabel, "Low", "eqLow");
    setupKnob(eqMidSlider, eqMidLabel, "Mid", "eqMid");
    setupKnob(eqHighSlider, eqHighLabel, "High", "eqHigh");

    //==========================================================================
    setupTitle(wtWarpTitle, "WT WARP");
    setupSmallLabel(wtWarpModeLabel, "Mode");
    setupCombo(wtWarpBox, "wtWarp",
        { "Off", "Bend", "Asym", "Quantize", "Smooth" });
    setupKnob(wtWarpAmtSlider, wtWarpAmtLabel, "Amt", "wtWarpAmt");
    
    setupTitle(warpTitle, "OSC WARP");
    setupKnob(pwmSlider, pwmLabel, "PWM", "pwmWidth");
    setupKnob(fmSlider, fmLabel, "FM", "fmAmount");
    setupCombo(syncBox, "hardSync", { "Off", "On" });
    setupSmallLabel(syncLabel, "Sync");

    setupTitle(lfo2Title, "LFO 2");
    setupKnob(lfo2RateSlider, lfo2RateLabel, "Rate", "lfo2Rate");
    setupKnob(lfo2DepthSlider, lfo2DepthLabel, "Depth", "lfo2Depth");
    setupCombo(lfo2ShapeBox, "lfo2Shape", { "Sine", "Tri", "Saw", "Square", "S&H" });
    setupCombo(lfo2SyncBox, "lfo2Sync",
        { "Free", "1/4", "1/4T", "1/8", "1/8T", "1/16", "1/16T", "1/32" });
    setupSmallLabel(lfo2ShapeLabel, "Shape");
    setupSmallLabel(lfo2SyncLabel, "Sync");

    setupTitle(modTitle, "MOD MATRIX");

    for (int i = 0; i < 4; ++i)
    {
        const juce::String idx = juce::String(i + 1);

        setupCombo(modSrcBox[i], "modSrc" + idx,
            { "Off", "LFO 1", "LFO 2", "Env Amp", "Env Filt",
              "ModWheel", "Velocity", "Keytrack", "Random", "Aftertouch",
              "MPE Timb", "MPE Press" });

        setupCombo(modDestBox[i], "modDest" + idx,
            { "Off", "WT Pos", "Cutoff", "Resonance",
              "Osc1 Lvl", "Osc2 Lvl", "Osc3 Lvl",
              "Osc1 Pch", "Osc2 Pch", "Osc3 Pch",
              "Volume", "Filt Drv" });

        modAmtSlider[i].setSliderStyle(juce::Slider::LinearHorizontal);
        modAmtSlider[i].setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        content.addAndMakeVisible(modAmtSlider[i]);

        sliderAttachments.push_back(
            std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                synthProcessor.parameters, "modAmt" + idx, modAmtSlider[i]));
    }

    //==========================================================================
    setupTitle(arpTitle, "ARP");
    setupSmallLabel(arpOnLabel, "On");
    setupCombo(arpOnBox, "arpOn", { "Off", "On" });
    setupSmallLabel(arpModeLabel, "Mode");
    setupCombo(arpModeBox, "arpMode", { "Up", "Down", "UpDn", "DnUp", "Random", "Order" });
    setupSmallLabel(arpRateLabel, "Rate");
    setupCombo(arpRateBox, "arpRate", { "1/4", "1/8", "1/8T", "1/16", "1/16T", "1/32" });
    setupKnob(arpOctSlider, arpOctLabel, "Oct", "arpOctaves");
    setupKnob(arpGateSlider, arpGateLabel, "Gate", "arpGate");

    setupSmallLabel(mpeOnLabel, "MPE");
    setupCombo(mpeOnBox, "mpeOn", { "Off", "On" });
    setupKnob(mpeBendSlider, mpeBendLabel, "Bend", "mpeBend");
    
    setupTitle(compTitle, "MASTER COMP");
    setupCombo(compOnBox, "compOn", { "Off", "On" });
    setupKnob(compThreshSlider, compThreshLabel, "Thresh", "compThresh");
    setupKnob(compRatioSlider, compRatioLabel, "Ratio", "compRatio");
    setupKnob(compAttackSlider, compAttackLabel, "Attack", "compAttack");
    setupKnob(compReleaseSlider, compReleaseLabel, "Release", "compRelease");

    //==========================================================================
    setupTitle(driveTitle, "DRIVE");
    setupKnob(fxDriveSlider, fxDriveLabel, "Drive", "fxDrive");

    setupTitle(chorusTitle, "CHORUS");
    setupKnob(chorusRateSlider, chorusRateLabel, "Rate", "chorusRate");
    setupKnob(chorusDepthSlider, chorusDepthLabel, "Depth", "chorusDepth");
    setupKnob(chorusMixSlider, chorusMixLabel, "Mix", "chorusMix");

    setupTitle(delayTitle, "DELAY");
    setupKnob(delayTimeSlider, delayTimeLabel, "Time", "delayTime");
    setupKnob(delayFeedbackSlider, delayFeedbackLabel, "Feedback", "delayFeedback");
    setupKnob(delayMixSlider, delayMixLabel, "Mix", "delayMix");

    setupTitle(reverbTitle, "REVERB");
    setupKnob(reverbSizeSlider, reverbSizeLabel, "Size", "reverbSize");
    setupKnob(reverbMixSlider, reverbMixLabel, "Mix", "reverbMix");

    //==========================================================================
    content.addAndMakeVisible(wtEditor);
    wtEditor.setVisible(false);
    
    setResizable(true, false);

    if (const auto* disp = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay())
    {
        const auto area = disp->userArea;

        const float s = juce::jmin(1.0f,
            (static_cast<float>(area.getHeight()) - 60.0f) / 1050.0f,
            (static_cast<float>(area.getWidth()) - 20.0f) / 1000.0f);

        setSize(juce::jmax(500, static_cast<int>(1000.0f * s)),
            juce::jmax(525, static_cast<int>(1050.0f * s)));
    }
    else
    {
        setSize(1000, 1050);
    }

    learnLabel.setFont(juce::Font(juce::FontOptions(10.0f)));
    learnLabel.setJustificationType(juce::Justification::centredLeft);
    learnLabel.setColour(juce::Label::textColourId, juce::Colour(0xffff4fa0));
    content.addAndMakeVisible(learnLabel);

    addMouseListener(this, true);

    startTimerHz(10);
}

StraticSynthAudioProcessorEditor::~StraticSynthAudioProcessorEditor()
{
    setLookAndFeel(nullptr);
}

void StraticSynthAudioProcessorEditor::mouseDown(const juce::MouseEvent& e)
{
    if (e.mods.isRightButtonDown())
    {
        auto it = learnTargets.find(e.originalComponent);

        if (it != learnTargets.end())
        {
            if (synthProcessor.isMidiLearnPending())
                synthProcessor.cancelMidiLearn();
            else
                synthProcessor.startMidiLearn(it->second);

            return;
        }
    }
}

void StraticSynthAudioProcessorEditor::timerCallback()
{
    const int count = synthProcessor.getNumPresets();

    juce::String sig;

    for (int i = 0; i < count; ++i)
        sig << synthProcessor.getPresetName(i) << "|";

    if (sig != lastPresetSig)
    {
        lastPresetSig = sig;
        lastPresetCount = count;

        presetBox.clear(juce::dontSendNotification);

        for (int i = 0; i < count; ++i)
            presetBox.addItem(synthProcessor.getPresetName(i), i + 1);
    }

    const int idx = synthProcessor.getCurrentPresetIndex();
    if (idx != lastPresetIndex)
    {
        lastPresetIndex = idx;
        presetBox.setSelectedId(idx + 1, juce::dontSendNotification);
    }

    const auto name = synthProcessor.getCurrentPresetName();
    if (name != lastPresetName)
    {
        lastPresetName = name;
        presetNameLabel.setText(name, juce::dontSendNotification);
    }

    logoPhase += 0.06f;
    outputMeter.setSampleRate(synthProcessor.getSampleRate());
    bgPanel.setPhase(logoPhase);
    bgPanel.repaint(0, 0, 1000, 50);
    bgPanel.repaint(936, 452, 60, 76);

    const auto sName = synthProcessor.getSampleName();
    if (sName != lastSampleName)
    {
        lastSampleName = sName;
        sampleTitle.setText(sName.isEmpty() ? "SAMPLE (no file)" : ("SAMPLE: " + sName),
            juce::dontSendNotification);
    }

    //==========================================================================
    const bool pending = synthProcessor.isMidiLearnPending();
    const auto maps = synthProcessor.getMappings();

    if (pending != lastLearnPending || static_cast<int>(maps.size()) != lastMapCount)
    {
        lastLearnPending = pending;
        lastMapCount = static_cast<int>(maps.size());

        if (pending)
            learnLabel.setText("MIDI LEARN: move a controller...\n(right-click a knob to cancel)",
                juce::dontSendNotification);
        else if (lastMapCount > 0)
            learnLabel.setText("MIDI mappings: " + juce::String(lastMapCount),
                juce::dontSendNotification);
        else
            learnLabel.setText("", juce::dontSendNotification);

        for (auto& kv : paramLabelMap)
        {
            bool mapped = false;

            for (const auto& m : maps)
            {
                if (m.first == kv.first)
                {
                    mapped = true;
                    break;
                }
            }

            kv.second->setColour(juce::Label::textColourId,
                mapped ? juce::Colour(0xffff4fa0)
                : juce::Colour(0xffc7cfdd));
        }
    }
}

void StraticSynthAudioProcessorEditor::paint(juce::Graphics& g)
{
    juce::ColourGradient bg(juce::Colour(0xff0b0e14), 0.0f, 0.0f,
        juce::Colour(0xff11151d), 0.0f,
        static_cast<float>(getHeight()), false);
    g.setGradientFill(bg);
    g.fillAll();
}

void StraticSynthAudioProcessorEditor::resized()
{
    const float s = juce::jmin(static_cast<float>(getWidth()) / 1000.0f,
        static_cast<float>(getHeight()) / 1050.0f);

    uiScale = s;

    content.setSize(1000, 1050);
    content.setTransform(juce::AffineTransform::scale(s));
    content.setTopLeftPosition((getWidth() - static_cast<int>(1000.0f * s)) / 2,
        (getHeight() - static_cast<int>(1050.0f * s)) / 2);

    //==========================================================================
    titleLabel.setBounds(12, 6, 200, 26);
    presetNameLabel.setBounds(215, 8, 330, 24);

    prevButton.setBounds(560, 8, 34, 24);
    presetBox.setBounds(600, 8, 220, 24);
    nextButton.setBounds(826, 8, 34, 24);
    saveButton.setBounds(866, 8, 60, 24);
    loadButton.setBounds(930, 8, 60, 24);

    wavetableDisplay.setBounds(10, 40, 620, 230);
    loadWtButton.setBounds(222, 46, 70, 18);
    resetWtButton.setBounds(296, 46, 70, 18);
    editWtButton.setBounds(370, 46, 70, 18);
    bankButton.setBounds(444, 46, 60, 18);

    wtEditor.setBounds(100, 120, 800, 620);
    envelopeDisplay.setBounds(640, 40, 350, 230);

    auto placeKnob = [](juce::Rectangle<int> area, juce::Label& label,
        juce::Slider& slider, int knobSize)
        {
            const int kx = area.getX() + (area.getWidth() - knobSize) / 2;
            slider.setBounds(kx, area.getY(), knobSize, knobSize);
            label.setBounds(area.getX(), area.getY() + knobSize + 4, area.getWidth(), 14);
        };

    //==========================================================================
    for (int i = 0; i < 3; ++i)
    {
        auto strip = juce::Rectangle<int>(14 + i * 206, 284, 198, 140);

        oscTitles[i].setBounds(strip.removeFromTop(16));
        strip.removeFromTop(2);
        oscBoxes[i].setBounds(strip.removeFromTop(24));
        strip.removeFromTop(8);

        if (i == 0)
        {
            placeKnob(strip, levelLabels[i], levelSliders[i], 72);
        }
        else
        {
            auto left = strip.removeFromLeft(96);
            strip.removeFromLeft(6);
            auto right = strip;

            placeKnob(left, levelLabels[i], levelSliders[i], 64);

            if (i == 1)
                placeKnob(right, auxLabels[0], auxSliders[0], 64);
            else
                placeKnob(right, auxLabels[1], auxSliders[1], 64);
        }
    }

    filterTitle.setBounds(644, 284, 180, 16);

    wtWarpTitle.setBounds(830, 284, 150, 16);
    wtWarpModeLabel.setBounds(836, 304, 90, 12);
    wtWarpBox.setBounds(836, 318, 90, 22);
    placeKnob({ 926, 304, 60, 60 }, wtWarpAmtLabel, wtWarpAmtSlider, 44);

    auto fRow = juce::Rectangle<int>(648, 308, 336, 90);
    auto f1 = fRow.removeFromLeft(80);
    fRow.removeFromLeft(5);
    auto f2 = fRow.removeFromLeft(80);
    placeKnob(f1, lfoRateLabel, lfoRateSlider, 62);
    placeKnob(f2, lfoDepthLabel, lfoDepthSlider, 62);

    //==========================================================================
    masterTitle.setBounds(16, 436, 300, 16);
    placeKnob({ 20, 458, 90, 84 }, wtPosLabel, wtPosSlider, 64);
    placeKnob({ 124, 458, 90, 84 }, volumeLabel, volumeSlider, 64);

    voiceModeLabel.setBounds(240, 458, 110, 16);
    voiceModeBox.setBounds(240, 478, 110, 24);
    placeKnob({ 360, 458, 90, 84 }, glideLabel, glideSlider, 64);

    uniTitle.setBounds(470, 436, 260, 16);
    placeKnob({ 474, 458, 84, 84 }, uniVoicesLabel, uniVoicesSlider, 64);
    placeKnob({ 562, 458, 84, 84 }, uniDetuneLabel, uniDetuneSlider, 64);
    placeKnob({ 650, 458, 84, 84 }, uniSpreadLabel, uniSpreadSlider, 64);

    sampleTitle.setBounds(740, 436, 180, 16);
    loadSampleButton.setBounds(924, 434, 62, 20);
    placeKnob({ 744, 452, 62, 66 }, sampleLevelLabel, sampleLevelSlider, 48);
    placeKnob({ 810, 452, 62, 66 }, sampleRootLabel, sampleRootSlider, 48);
    placeKnob({ 876, 452, 62, 66 }, sampleStartLabel, sampleStartSlider, 48);

    sampleModeLabel.setBounds(744, 520, 70, 12);
    sampleModeBox.setBounds(744, 532, 70, 20);
    sampleRevLabel.setBounds(820, 520, 60, 12);
    sampleRevBox.setBounds(820, 532, 60, 20);

    hintLabel.setBounds(0, 0, 0, 0);

    //==========================================================================
    filterPanelTitle.setBounds(16, 566, 100, 16);
    filterTypeBox.setBounds(16, 584, 70, 24);

    placeKnob({ 100, 584, 70, 70 }, cutoffLabel, cutoffSlider, 52);
    placeKnob({ 170, 584, 70, 70 }, resonanceLabel, resonanceSlider, 52);
    placeKnob({ 240, 584, 70, 70 }, filterDriveLabel, filterDriveSlider, 52);
    placeKnob({ 310, 584, 70, 70 }, keytrackLabel, keytrackSlider, 52);
    placeKnob({ 380, 584, 70, 70 }, fenvAmtLabel, fenvAmtSlider, 52);
    placeKnob({ 450, 584, 70, 70 }, fenvALabel, fenvASlider, 52);
    placeKnob({ 520, 584, 70, 70 }, fenvDLabel, fenvDSlider, 52);
    placeKnob({ 590, 584, 70, 70 }, fenvSLabel, fenvSSlider, 52);
    placeKnob({ 660, 584, 70, 70 }, fenvRLabel, fenvRSlider, 52);

    eqTitle.setBounds(740, 566, 100, 16);
    placeKnob({ 744, 584, 70, 70 }, eqLowLabel, eqLowSlider, 52);
    placeKnob({ 814, 584, 70, 70 }, eqMidLabel, eqMidSlider, 52);
    placeKnob({ 884, 584, 70, 70 }, eqHighLabel, eqHighSlider, 52);

    //==========================================================================
    warpTitle.setBounds(760, 670, 120, 16);
    placeKnob({ 764, 688, 70, 70 }, pwmLabel, pwmSlider, 52);
    placeKnob({ 834, 688, 70, 70 }, fmLabel, fmSlider, 52);
    syncLabel.setBounds(908, 688, 70, 14);
    syncBox.setBounds(908, 704, 70, 22);

    lfo2Title.setBounds(16, 670, 100, 16);
    placeKnob({ 20, 688, 70, 70 }, lfo2RateLabel, lfo2RateSlider, 52);
    placeKnob({ 90, 688, 70, 70 }, lfo2DepthLabel, lfo2DepthSlider, 52);

    lfo2ShapeLabel.setBounds(170, 688, 90, 14);
    lfo2ShapeBox.setBounds(170, 704, 90, 22);
    lfo2SyncLabel.setBounds(270, 688, 90, 14);
    lfo2SyncBox.setBounds(270, 704, 90, 22);

    modTitle.setBounds(380, 670, 140, 16);

    for (int i = 0; i < 4; ++i)
    {
        const int row = 690 + i * 26;

        modSrcBox[i].setBounds(380, row, 100, 22);
        modDestBox[i].setBounds(488, row, 100, 22);
        modAmtSlider[i].setBounds(600, row + 2, 150, 18);
    }

    //==========================================================================
    compTitle.setBounds(16, 946, 160, 16);
    compOnBox.setBounds(16, 968, 64, 22);

    arpTitle.setBounds(380, 946, 120, 16);

    arpOnLabel.setBounds(386, 964, 50, 12);
    arpOnBox.setBounds(386, 978, 50, 22);

    arpModeLabel.setBounds(442, 964, 90, 12);
    arpModeBox.setBounds(442, 978, 90, 22);

    arpRateLabel.setBounds(538, 964, 80, 12);
    arpRateBox.setBounds(538, 978, 80, 22);

    placeKnob({ 626, 962, 60, 60 }, arpOctLabel, arpOctSlider, 44);
    placeKnob({ 692, 962, 60, 60 }, arpGateLabel, arpGateSlider, 44);
    learnLabel.setBounds(760, 946, 220, 18);

    mpeOnLabel.setBounds(760, 966, 44, 12);
    mpeOnBox.setBounds(760, 980, 44, 22);
    placeKnob({ 812, 964, 60, 60 }, mpeBendLabel, mpeBendSlider, 44);

    placeKnob({ 96, 968, 64, 64 }, compThreshLabel, compThreshSlider, 48);
    placeKnob({ 164, 968, 64, 64 }, compRatioLabel, compRatioSlider, 48);
    placeKnob({ 232, 968, 64, 64 }, compAttackLabel, compAttackSlider, 48);
    placeKnob({ 300, 968, 64, 64 }, compReleaseLabel, compReleaseSlider, 48);

    //==========================================================================
    driveTitle.setBounds(16, 806, 100, 16);
    placeKnob({ 20, 824, 90, 84 }, fxDriveLabel, fxDriveSlider, 64);

    chorusTitle.setBounds(140, 806, 260, 16);
    placeKnob({ 144, 824, 84, 84 }, chorusRateLabel, chorusRateSlider, 64);
    placeKnob({ 232, 824, 84, 84 }, chorusDepthLabel, chorusDepthSlider, 64);
    placeKnob({ 320, 824, 84, 84 }, chorusMixLabel, chorusMixSlider, 64);

    delayTitle.setBounds(420, 806, 260, 16);
    placeKnob({ 424, 824, 84, 84 }, delayTimeLabel, delayTimeSlider, 64);
    placeKnob({ 512, 824, 84, 84 }, delayFeedbackLabel, delayFeedbackSlider, 64);
    placeKnob({ 600, 824, 84, 84 }, delayMixLabel, delayMixSlider, 64);

    reverbTitle.setBounds(700, 806, 260, 16);
    placeKnob({ 704, 824, 84, 84 }, reverbSizeLabel, reverbSizeSlider, 64);
    placeKnob({ 792, 824, 84, 84 }, reverbMixLabel, reverbMixSlider, 64);

    outputMeter.setBounds(884, 806, 100, 120);
}