#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

//==============================================================================
struct StraticLookAndFeel : public juce::LookAndFeel_V4
{
    void drawRotarySlider(juce::Graphics& g,
        int x, int y, int width, int height,
        float sliderPosProportional,
        float rotaryStartAngle,
        float rotaryEndAngle,
        juce::Slider& slider) override;

    void drawComboBox(juce::Graphics& g,
        int width, int height,
        bool isButtonDown,
        int buttonX, int buttonY, int buttonW, int buttonH,
        juce::ComboBox& box) override;

    void drawButtonBackground(juce::Graphics& g,
        juce::Button& button,
        const juce::Colour& backgroundColour,
        bool shouldDrawButtonAsHighlighted,
        bool shouldDrawButtonAsDown) override;
};

//==============================================================================
class WavetableDisplay : public juce::Component,
    public juce::Timer
{
public:
    WavetableDisplay(juce::AudioProcessorValueTreeState& state,
        std::atomic<WavetableData*>* source);

    void paint(juce::Graphics& g) override;
    void timerCallback() override;

    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseDoubleClick(const juce::MouseEvent& e) override;
    void mouseWheelMove(const juce::MouseEvent& e,
        const juce::MouseWheelDetails& wheel) override;

private:
    void dragTo(const juce::MouseEvent& e);
    const WavetableData& activeTable() const;

    std::atomic<float>* wtAtomic = nullptr;
    juce::RangedAudioParameter* wtParam = nullptr;
    std::atomic<WavetableData*>* wtSource = nullptr;

    WavetableData builtIn;

    bool stackView = true;

    float lastValue = -1.0f;
    void* lastPtr = nullptr;
    bool lastStack = true;
};

//==============================================================================
class EnvelopeDisplay : public juce::Component,
    public juce::Timer
{
public:
    explicit EnvelopeDisplay(juce::AudioProcessorValueTreeState& state);

    void paint(juce::Graphics& g) override;
    void timerCallback() override;

    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;

private:
    struct EnvLayout
    {
        float x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0;
        float yTop = 0, yBot = 0, yS = 0;
        float maxA = 0, maxD = 0, maxR = 0;
    };

    EnvLayout computeLayout() const;
    void setParam(juce::RangedAudioParameter* p, float value);

    std::atomic<float>* aAtomic = nullptr;
    std::atomic<float>* dAtomic = nullptr;
    std::atomic<float>* sAtomic = nullptr;
    std::atomic<float>* rAtomic = nullptr;

    juce::RangedAudioParameter* aParam = nullptr;
    juce::RangedAudioParameter* dParam = nullptr;
    juce::RangedAudioParameter* sParam = nullptr;
    juce::RangedAudioParameter* rParam = nullptr;

    int activeHandle = -1;

    float lastSnapshot[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
};

//==============================================================================
class OutputMeter : public juce::Component,
    public juce::Timer
{
public:
    explicit OutputMeter(OutputMonitor* m)
        : mon(m)
    {
        bufL.resize(1024, 0.0f);
        bufR.resize(1024, 0.0f);
        fftRe.resize(1024, 0.0f);
        fftIm.resize(1024, 0.0f);
        win.resize(1024);
        bars.assign(48, 0.0f);

        for (int i = 0; i < 1024; ++i)
            win[i] = 0.5f - 0.5f * std::cos(
                2.0f * juce::MathConstants<float>::pi * static_cast<float>(i) / 1023.0f);

        startTimerHz(30);
    }

    void setSampleRate(float hz) { sr = hz > 1000.0f ? hz : 44100.0f; }

    void timerCallback() override
    {
        if (mon == nullptr)
            return;

        const float pL = mon->takePeakL();
        const float pR = mon->takePeakR();

        levelL = juce::jmax(pL, levelL * 0.80f);
        levelR = juce::jmax(pR, levelR * 0.80f);

        holdL = juce::jmax(holdL * 0.995f, levelL);
        holdR = juce::jmax(holdR * 0.995f, levelR);

        mon->copyScope(bufL, bufR);

        computeSpectrum();

        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();

        g.setColour(juce::Colour(0xff12151c));
        g.fillRoundedRectangle(bounds, 6.0f);
        g.setColour(juce::Colour(0xff2a3140));
        g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.0f);

        g.setColour(juce::Colour(0xff9fb2ff));
        g.setFont(juce::Font(juce::FontOptions(10.0f)));
        g.drawText("OUTPUT", 6, 3, 60, 12, juce::Justification::left);

        auto scope = bounds.withTrimmedTop(16.0f)
            .withTrimmedRight(24.0f)
            .reduced(4.0f);

        auto vuArea = juce::Rectangle<float>(bounds.getRight() - 20.0f,
            scope.getY(), 16.0f, scope.getHeight());

        g.setColour(juce::Colour(0xff0c0f16));
        g.fillRoundedRectangle(scope, 4.0f);

        g.setColour(juce::Colour(0x14ffffff));
        g.drawHorizontalLine(static_cast<int>(scope.getCentreY()),
            scope.getX(), scope.getRight());

        const int numBars = static_cast<int>(bars.size());
        const float bw = scope.getWidth() / static_cast<float>(numBars);

        for (int b = 0; b < numBars; ++b)
        {
            const float h = bars[static_cast<size_t>(b)] * scope.getHeight() * 0.5f;

            if (h < 1.0f)
                continue;

            const float x = scope.getX() + static_cast<float>(b) * bw;

            juce::ColourGradient barGrad(juce::Colour(0x6600e5ff), x, scope.getCentreY(),
                juce::Colour(0x1100e5ff), x,
                scope.getCentreY() - h, false);

            g.setGradientFill(barGrad);
            g.fillRect(x + 0.5f, scope.getCentreY() - h, bw - 1.0f, h * 2.0f);
        }

        auto drawWave = [&](const std::vector<float>& buf, juce::Colour col)
            {
                juce::Path path;
                const int n = static_cast<int>(buf.size());
                const int w = static_cast<int>(scope.getWidth());
                if (w < 2) return;

                const float step = static_cast<float>(n) / static_cast<float>(w);

                for (int x = 0; x < w; ++x)
                {
                    const int idx = juce::jlimit(0, n - 1,
                        static_cast<int>(static_cast<float>(x) * step));
                    const float v = juce::jlimit(-1.0f, 1.0f, buf[static_cast<size_t>(idx)]);

                    const float px = scope.getX() + static_cast<float>(x);
                    const float py = scope.getCentreY() - v * scope.getHeight() * 0.45f;

                    if (x == 0) path.startNewSubPath(px, py);
                    else path.lineTo(px, py);
                }

                g.setColour(col);
                g.strokePath(path, juce::PathStrokeType(1.2f));
            };

        drawWave(bufR, juce::Colour(0xaaff4fa0));
        drawWave(bufL, juce::Colour(0xff00e5ff));

        auto drawVu = [&](float level, float hold, float x)
            {
                const float h = vuArea.getHeight();

                g.setColour(juce::Colour(0xff0c0f16));
                g.fillRoundedRectangle(juce::Rectangle<float>(x, vuArea.getY(), 6.0f, h), 2.0f);

                auto toPos = [](float lin)
                    {
                        if (lin <= 0.00001f) return 0.0f;
                        const float db = juce::Decibels::gainToDecibels(lin);
                        return juce::jlimit(0.0f, 1.0f, (db + 48.0f) / 48.0f);
                    };

                const float fillH = h * toPos(level);

                if (fillH > 0.5f)
                {
                    juce::ColourGradient grad(juce::Colour(0xff00e5ff), x, vuArea.getBottom(),
                        juce::Colour(0xffff3b3b), x, vuArea.getY(), false);
                    g.setGradientFill(grad);
                    g.fillRoundedRectangle(
                        juce::Rectangle<float>(x, vuArea.getBottom() - fillH, 6.0f, fillH), 2.0f);
                }

                g.setColour(juce::Colours::white);
                g.fillRect(x, vuArea.getBottom() - h * toPos(hold) - 1.0f, 6.0f, 1.5f);

                if (hold > 0.99f)
                {
                    g.setColour(juce::Colour(0xffff3b3b));
                    g.fillRoundedRectangle(
                        juce::Rectangle<float>(x - 1.0f, vuArea.getY() - 3.0f, 8.0f, 2.0f), 1.0f);
                }
            };

        drawVu(levelL, holdL, vuArea.getX());
        drawVu(levelR, holdR, vuArea.getX() + 9.0f);
    }

private:
    void computeSpectrum();

    OutputMonitor* mon = nullptr;
    float levelL = 0.0f, levelR = 0.0f;
    float holdL = 0.0f, holdR = 0.0f;
    float sr = 44100.0f;
    std::vector<float> bufL, bufR;
    std::vector<float> fftRe, fftIm, win, bars;
};

//==============================================================================
class WavetableEditor : public juce::Component
{
public:
    explicit WavetableEditor(StraticSynthAudioProcessor& p)
        : proc(p)
    {
        auto mk = [this](juce::TextButton& b, const juce::String& t)
            {
                b.setButtonText(t);
                addAndMakeVisible(b);
            };

        mk(sineBtn, "Sine"); mk(sawBtn, "Saw"); mk(sqrBtn, "Square");
        mk(smoothBtn, "Smooth"); mk(normBtn, "Normalize");
        mk(addBtn, "Add Frame"); mk(delBtn, "Del Frame"); mk(initBtn, "Init 8");
        mk(applyBtn, "Apply"); mk(closeBtn, "Close");

        sineBtn.onClick = [this] { fill(0); };
        sawBtn.onClick = [this] { fill(1); };
        sqrBtn.onClick = [this] { fill(2); };
        smoothBtn.onClick = [this] { smooth(); };
        normBtn.onClick = [this] { normalize(); };
        addBtn.onClick = [this] { addFrame(); };
        delBtn.onClick = [this] { delFrame(); };
        initBtn.onClick = [this] { work = WavetableData(); currentFrame = 0; repaint(); };
        applyBtn.onClick = [this] { proc.applyWavetable(work); setVisible(false); };
        closeBtn.onClick = [this] { setVisible(false); };

        setVisible(false);
    }

    void open(const WavetableData* src)
    {
        if (src != nullptr && !src->frames.empty())
        {
            work.frames = src->frames;
            work.tableSize = src->tableSize;
            work.numFrames = src->numFrames;
        }
        else
        {
            work = WavetableData();
        }

        currentFrame = 0;
        setVisible(true);
        repaint();
    }

    void resized() override
    {
        auto r = getLocalBounds().reduced(10);

        auto row2 = r.removeFromBottom(30);
        auto row1 = r.removeFromBottom(34);
        strip = r.removeFromBottom(70);
        r.removeFromBottom(6);
        wave = r;

        int x = 0;
        auto place = [&](juce::TextButton& b, int w)
            {
                b.setBounds(row1.getX() + x, row1.getY(), w, 24);
                x += w + 8;
            };

        place(sineBtn, 70); place(sawBtn, 70); place(sqrBtn, 70);
        place(smoothBtn, 80); place(normBtn, 90);

        x = 0;
        auto place2 = [&](juce::TextButton& b, int w)
            {
                b.setBounds(row2.getX() + x, row2.getY(), w, 24);
                x += w + 8;
            };

        place2(addBtn, 90); place2(delBtn, 90); place2(initBtn, 70);

        closeBtn.setBounds(row2.getRight() - 80, row2.getY(), 80, 24);
        applyBtn.setBounds(row2.getRight() - 170, row2.getY(), 80, 24);
    }

    void paint(juce::Graphics& g) override
    {
        const auto b = getLocalBounds().toFloat();

        g.setColour(juce::Colour(0xdd10141c));
        g.fillRoundedRectangle(b, 12.0f);
        g.setColour(juce::Colour(0xff00e5ff));
        g.drawRoundedRectangle(b.reduced(0.5f), 12.0f, 1.5f);

        g.setColour(juce::Colour(0xff9fb2ff));
        g.setFont(juce::Font(juce::FontOptions(14.0f)));
        g.drawText("WAVETABLE EDITOR   frames: " + juce::String(work.numFrames) +
            "   editing: " + juce::String(currentFrame + 1),
            14, 8, 500, 20, juce::Justification::left);

        g.setColour(juce::Colour(0xff0c0f16));
        g.fillRoundedRectangle(wave.toFloat(), 6.0f);

        g.setColour(juce::Colour(0x14ffffff));
        g.drawHorizontalLine(static_cast<int>(wave.getCentreY()),
            wave.getX(), wave.getRight());

        if (currentFrame < static_cast<int>(work.frames.size()))
        {
            const auto& fr = work.frames[static_cast<size_t>(currentFrame)];
            const int n = static_cast<int>(fr.size());

            juce::Path path;
            for (int i = 0; i < wave.getWidth(); ++i)
            {
                const int idx = juce::jlimit(0, n - 1,
                    static_cast<int>(static_cast<float>(i) * static_cast<float>(n) /
                        static_cast<float>(juce::jmax(1, wave.getWidth()))));
                const float v = fr[static_cast<size_t>(idx)];
                const float py = wave.getCentreY() - v * wave.getHeight() * 0.45f;

                if (i == 0) path.startNewSubPath(static_cast<float>(wave.getX() + i), py);
                else path.lineTo(static_cast<float>(wave.getX() + i), py);
            }

            g.setColour(juce::Colour(0xff00e5ff));
            g.strokePath(path, juce::PathStrokeType(2.0f));
        }

        const int F = work.numFrames;

        if (F > 0)
        {
            const float cw = static_cast<float>(strip.getWidth()) / static_cast<float>(F);

            for (int f = 0; f < F; ++f)
            {
                auto cell = juce::Rectangle<float>(strip.getX() + cw * static_cast<float>(f),
                    strip.getY(), cw, strip.getHeight());

                g.setColour(f == currentFrame ? juce::Colour(0x5500e5ff)
                    : juce::Colour(0x22ffffff));
                g.drawRoundedRectangle(cell.reduced(1.0f), 3.0f, 1.0f);

                if (f < static_cast<int>(work.frames.size()))
                {
                    const auto& fr = work.frames[static_cast<size_t>(f)];
                    const int n = static_cast<int>(fr.size());

                    juce::Path mp;
                    const int mw = 40;

                    for (int i = 0; i < mw; ++i)
                    {
                        const int idx = juce::jlimit(0, n - 1, i * n / mw);
                        const float v = fr[static_cast<size_t>(idx)];
                        const float px = cell.getX() + 2.0f +
                            (cell.getWidth() - 4.0f) * static_cast<float>(i) / static_cast<float>(mw - 1);
                        const float py = cell.getCentreY() - v * cell.getHeight() * 0.4f;

                        if (i == 0) mp.startNewSubPath(px, py);
                        else mp.lineTo(px, py);
                    }

                    g.setColour(f == currentFrame ? juce::Colour(0xff00e5ff)
                        : juce::Colour(0x887f8ba3));
                    g.strokePath(mp, juce::PathStrokeType(1.0f));
                }
            }
        }
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto p = e.getPosition();

        if (strip.contains(p))
        {
            const int F = work.numFrames;
            if (F > 0)
            {
                currentFrame = juce::jlimit(0, F - 1,
                    static_cast<int>((static_cast<float>(p.x) - strip.getX()) /
                        static_cast<float>(juce::jmax(1, strip.getWidth())) *
                        static_cast<float>(F)));
                repaint();
            }
            return;
        }

        if (wave.contains(p))
        {
            lastPt = p;
            setSampleAt(p);
            repaint();
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        const auto p = e.getPosition();

        if (wave.contains(p) || wave.contains(lastPt))
        {
            const int dx = std::abs(p.x - lastPt.x);
            const int dy = std::abs(p.y - lastPt.y);
            const int steps = juce::jmax(1, dx + dy);

            for (int i = 0; i <= steps; ++i)
                setSampleAt(lastPt + (p - lastPt) * i / steps);

            repaint();
        }

        lastPt = p;
    }

private:
    void setSampleAt(juce::Point<int> p)
    {
        if (currentFrame >= static_cast<int>(work.frames.size()))
            return;

        auto& fr = work.frames[static_cast<size_t>(currentFrame)];
        const int n = static_cast<int>(fr.size());

        if (n < 2 || wave.getWidth() < 2)
            return;

        const int idx = juce::jlimit(0, n - 1,
            static_cast<int>((static_cast<float>(p.x - wave.getX()) /
                static_cast<float>(wave.getWidth())) * static_cast<float>(n)));

        const float v = juce::jlimit(-1.0f, 1.0f,
            1.0f - 2.0f * (static_cast<float>(p.y - wave.getY()) /
                static_cast<float>(juce::jmax(1, wave.getHeight()))));

        fr[static_cast<size_t>(idx)] = v;
    }

    void fill(int shape)
    {
        if (currentFrame >= static_cast<int>(work.frames.size()))
            return;

        auto& fr = work.frames[static_cast<size_t>(currentFrame)];
        const int n = static_cast<int>(fr.size());

        for (int i = 0; i < n; ++i)
        {
            const float p = static_cast<float>(i) / static_cast<float>(juce::jmax(1, n));

            if (shape == 0) fr[static_cast<size_t>(i)] = std::sin(juce::MathConstants<float>::twoPi * p);
            else if (shape == 1) fr[static_cast<size_t>(i)] = 2.0f * p - 1.0f;
            else fr[static_cast<size_t>(i)] = p < 0.5f ? 1.0f : -1.0f;
        }

        repaint();
    }

    void smooth()
    {
        if (currentFrame >= static_cast<int>(work.frames.size()))
            return;

        auto& fr = work.frames[static_cast<size_t>(currentFrame)];
        const int n = static_cast<int>(fr.size());
        std::vector<float> tmp = fr;

        for (int i = 0; i < n; ++i)
            fr[static_cast<size_t>(i)] =
            (tmp[static_cast<size_t>((i - 1 + n) % n)] + tmp[static_cast<size_t>(i)] +
                tmp[static_cast<size_t>((i + 1) % n)]) / 3.0f;

        repaint();
    }

    void normalize()
    {
        if (currentFrame >= static_cast<int>(work.frames.size()))
            return;

        auto& fr = work.frames[static_cast<size_t>(currentFrame)];
        float peak = 0.0f;
        for (float v : fr) peak = juce::jmax(peak, std::abs(v));

        if (peak > 0.0001f)
            for (auto& v : fr) v /= peak;

        repaint();
    }

    void addFrame()
    {
        if (work.numFrames >= 64 || work.frames.empty())
            return;

        auto copy = work.frames[static_cast<size_t>(
            juce::jlimit(0, static_cast<int>(work.frames.size()) - 1, currentFrame))];

        work.frames.insert(work.frames.begin() + currentFrame + 1, copy);
        work.numFrames = static_cast<int>(work.frames.size());
        ++currentFrame;
        repaint();
    }

    void delFrame()
    {
        if (work.numFrames <= 2)
            return;

        work.frames.erase(work.frames.begin() + currentFrame);
        work.numFrames = static_cast<int>(work.frames.size());
        currentFrame = juce::jlimit(0, work.numFrames - 1, currentFrame);
        repaint();
    }

    StraticSynthAudioProcessor& proc;

    WavetableData work;
    int currentFrame = 0;
    juce::Point<int> lastPt;

    juce::Rectangle<int> wave, strip;

    juce::TextButton sineBtn, sawBtn, sqrBtn, smoothBtn, normBtn;
    juce::TextButton addBtn, delBtn, initBtn, applyBtn, closeBtn;
};


//==============================================================================
class UIBackground : public juce::Component
{
public:
    void setPhase(float p) { phase = p; }

    void setSources(std::atomic<float>* mw, std::atomic<float>* at)
    {
        mwSrc = mw;
        atSrc = at;
    }

    void paint(juce::Graphics& g) override
    {
        juce::ColourGradient glow(juce::Colour(0x1400e5ff), 0.0f, 0.0f,
            juce::Colour(0x0000e5ff), 0.0f, 120.0f, false);
        g.setGradientFill(glow);
        g.fillRect(0, 0, 1000, 120);

        //======================================================================
        const float pulse = 0.5f + 0.5f * std::sin(phase);

        g.setColour(juce::Colour(0x00e5ff).withAlpha(0.10f + 0.12f * pulse));
        g.fillRoundedRectangle(8.0f, 4.0f, 208.0f, 30.0f, 8.0f);

        const float sweepX = 8.0f + std::fmod(phase * 40.0f, 260.0f);

        juce::ColourGradient sweep(juce::Colour(0x0000e5ff), sweepX - 30.0f, 0.0f,
            juce::Colour(0x4400e5ff), sweepX, 0.0f, false);
        g.setGradientFill(sweep);
        g.fillRoundedRectangle(juce::Rectangle<float>(sweepX - 30.0f, 4.0f, 30.0f, 30.0f), 6.0f);

        g.setColour(juce::Colour(0xff00e5ff));
        g.fillRect(8.0f, 34.0f, 40.0f + 160.0f * pulse, 1.5f);

        g.setColour(juce::Colour(0x6600e5ff));
        g.drawHorizontalLine(36, 0.0f, 1000.0f);

        //======================================================================
        auto panel = [&](juce::Rectangle<int> r)
            {
                const auto rf = r.toFloat();
                juce::ColourGradient pg(juce::Colour(0xff161b25), rf.getX(), rf.getY(),
                    juce::Colour(0xff10141c), rf.getX(), rf.getBottom(), false);
                g.setGradientFill(pg);
                g.fillRoundedRectangle(rf, 10.0f);
                g.setColour(juce::Colour(0xff2a3140));
                g.drawRoundedRectangle(rf.reduced(0.5f), 10.0f, 1.0f);
                g.setColour(juce::Colour(0x14ffffff));
                g.drawRoundedRectangle(rf.reduced(1.5f), 9.0f, 1.0f);
            };

        panel({ 10, 278, 624, 148 });
        panel({ 640, 278, 350, 148 });
        panel({ 10, 430, 980, 122 });
        panel({ 10, 560, 980, 96 });
        panel({ 10, 664, 980, 128 });
        panel({ 10, 800, 980, 130 });
        panel({ 10, 940, 980, 100 });

        auto accent = [&](float ax, float ay)
            {
                g.setColour(juce::Colour(0xaa00e5ff));
                g.fillRect(ax, ay, 26.0f, 2.0f);
                g.setColour(juce::Colour(0x3300e5ff));
                g.fillRect(ax + 26.0f, ay, 60.0f, 2.0f);
            };

        for (int i = 0; i < 3; ++i)
            accent(14.0f + static_cast<float>(i) * 206.0f, 302.0f);
        accent(644.0f, 302.0f);
        accent(16.0f, 454.0f);
        accent(470.0f, 454.0f);
        accent(740.0f, 454.0f);
        accent(16.0f, 686.0f);
        accent(16.0f, 824.0f);
        accent(16.0f, 964.0f);
        accent(380.0f, 964.0f);

        //======================================================================
        // индикаторы ModWheel / Aftertouch
        if (mwSrc != nullptr)
        {
            const float mw = mwSrc->load();
            const float at = atSrc != nullptr ? atSrc->load() : 0.0f;

            auto bar = [&](float x, float v, const char* name)
                {
                    g.setColour(juce::Colour(0xff0c0f16));
                    g.fillRoundedRectangle(juce::Rectangle<float>(x, 458.0f, 10.0f, 50.0f), 3.0f);

                    const float h = 50.0f * juce::jlimit(0.0f, 1.0f, v);

                    if (h > 0.5f)
                    {
                        g.setColour(juce::Colour(0xff00e5ff));
                        g.fillRoundedRectangle(
                            juce::Rectangle<float>(x, 458.0f + 50.0f - h, 10.0f, h), 3.0f);
                    }

                    g.setColour(juce::Colour(0xff8b93a7));
                    g.setFont(juce::Font(juce::FontOptions(8.0f)));
                    g.drawText(name, static_cast<int>(x) - 4, 510, 18, 10,
                        juce::Justification::centred);
                };

            bar(944.0f, mw, "MW");
            bar(962.0f, at, "AT");
        }
    }

private:
    float phase = 0.0f;
    std::atomic<float>* mwSrc = nullptr;
    std::atomic<float>* atSrc = nullptr;
};

//==============================================================================
class StraticSynthAudioProcessorEditor : public juce::AudioProcessorEditor,
    public juce::Timer
{
public:
    explicit StraticSynthAudioProcessorEditor(StraticSynthAudioProcessor& processor);
    ~StraticSynthAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;
    void mouseDown(const juce::MouseEvent& e) override;

private:
    StraticSynthAudioProcessor& synthProcessor;

    juce::Component content;
    UIBackground bgPanel;
    float uiScale = 1.0f;
    float logoPhase = 0.0f;

    WavetableDisplay wavetableDisplay;
    EnvelopeDisplay envelopeDisplay;
    OutputMeter outputMeter;

    StraticLookAndFeel look;

    juce::Label titleLabel;
    juce::Label presetNameLabel;
    juce::Label hintLabel;

    juce::ComboBox presetBox;
    juce::TextButton prevButton{ "<" };
    juce::TextButton nextButton{ ">" };
    juce::TextButton saveButton{ "Save" };
    juce::TextButton loadButton{ "Load" };

    int lastPresetIndex = -2;
    juce::String lastPresetName;
    int lastPresetCount = -1;
    juce::String lastPresetSig;

    juce::Label oscTitles[3];
    juce::ComboBox oscBoxes[3];
    juce::Slider levelSliders[3];
    juce::Label levelLabels[3];
    juce::Slider auxSliders[2];
    juce::Label auxLabels[2];

    juce::Label filterTitle;
    juce::Slider lfoRateSlider, lfoDepthSlider;
    juce::Label lfoRateLabel, lfoDepthLabel;

    juce::Label masterTitle;
    juce::Slider wtPosSlider, volumeSlider, glideSlider;
    juce::Label wtPosLabel, volumeLabel, glideLabel;
    juce::ComboBox voiceModeBox;
    juce::Label voiceModeLabel;

    juce::Label uniTitle;
    juce::Slider uniVoicesSlider, uniDetuneSlider, uniSpreadSlider;
    juce::Label uniVoicesLabel, uniDetuneLabel, uniSpreadLabel;

    juce::Label sampleTitle;
    juce::Slider sampleLevelSlider, sampleRootSlider, sampleStartSlider;
    juce::Label sampleLevelLabel, sampleRootLabel, sampleStartLabel;
    juce::ComboBox sampleModeBox, sampleRevBox;
    juce::Label sampleModeLabel, sampleRevLabel;
    juce::TextButton loadSampleButton{ "Load" };
    juce::String lastSampleName;

    juce::TextButton loadWtButton{ "WT Load" };
    juce::TextButton resetWtButton{ "WT Int" };
    juce::TextButton editWtButton{ "Edit" };
    juce::TextButton bankButton{ "Bank" };
    WavetableEditor wtEditor;

    juce::Label filterPanelTitle;
    juce::ComboBox filterTypeBox;
    juce::Slider cutoffSlider, resonanceSlider;
    juce::Slider filterDriveSlider, keytrackSlider;
    juce::Slider fenvAmtSlider, fenvASlider, fenvDSlider, fenvSSlider, fenvRSlider;
    juce::Label cutoffLabel, resonanceLabel;
    juce::Label filterDriveLabel, keytrackLabel;
    juce::Label fenvAmtLabel, fenvALabel, fenvDLabel, fenvSLabel, fenvRLabel;

    juce::Label eqTitle;
    juce::Slider eqLowSlider, eqMidSlider, eqHighSlider;
    juce::Label eqLowLabel, eqMidLabel, eqHighLabel;

    juce::Label wtWarpTitle;
    juce::ComboBox wtWarpBox;
    juce::Label wtWarpModeLabel;
    juce::Slider wtWarpAmtSlider;
    juce::Label wtWarpAmtLabel;
    
    juce::Label warpTitle;
    juce::Slider pwmSlider, fmSlider;
    juce::Label pwmLabel, fmLabel;
    juce::ComboBox syncBox;
    juce::Label syncLabel;

    juce::Label lfo2Title;
    juce::Slider lfo2RateSlider, lfo2DepthSlider;
    juce::ComboBox lfo2ShapeBox, lfo2SyncBox;
    juce::Label lfo2RateLabel, lfo2DepthLabel, lfo2ShapeLabel, lfo2SyncLabel;

    juce::Label modTitle;
    juce::ComboBox modSrcBox[4];
    juce::ComboBox modDestBox[4];
    juce::Slider modAmtSlider[4];

    juce::Label arpTitle;
    juce::ComboBox arpOnBox, arpModeBox, arpRateBox;
    juce::Label arpOnLabel, arpModeLabel, arpRateLabel;
    juce::Slider arpOctSlider, arpGateSlider;
    juce::Label arpOctLabel, arpGateLabel;
    juce::ComboBox mpeOnBox;
    juce::Label mpeOnLabel;
    juce::Slider mpeBendSlider;
    juce::Label mpeBendLabel;
    
    juce::Label learnLabel;
    std::map<juce::Component*, juce::String> learnTargets;
    std::map<juce::String, juce::Label*> paramLabelMap;
    bool lastLearnPending = false;
    int lastMapCount = -1;
    
    juce::Label compTitle;
    juce::ComboBox compOnBox;
    juce::Slider compThreshSlider, compRatioSlider, compAttackSlider, compReleaseSlider;
    juce::Label compThreshLabel, compRatioLabel, compAttackLabel, compReleaseLabel;

    juce::Label driveTitle, chorusTitle, delayTitle, reverbTitle;
    juce::Slider fxDriveSlider;
    juce::Label fxDriveLabel;
    juce::Slider chorusRateSlider, chorusDepthSlider, chorusMixSlider;
    juce::Label chorusRateLabel, chorusDepthLabel, chorusMixLabel;
    juce::Slider delayTimeSlider, delayFeedbackSlider, delayMixSlider;
    juce::Label delayTimeLabel, delayFeedbackLabel, delayMixLabel;
    juce::Slider reverbSizeSlider, reverbMixSlider;
    juce::Label reverbSizeLabel, reverbMixLabel;

    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
};