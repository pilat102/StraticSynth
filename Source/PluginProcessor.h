#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

//==============================================================================
struct SampleData
{
    juce::AudioBuffer<float> buffer;
    double sampleRate = 44100.0;
};

//==============================================================================
struct WavetableData
{
    std::vector<std::vector<float>> frames;

    int tableSize = 1024;
    int numFrames = 8;

    WavetableData()
    {
        generate();
    }

    void generate()
    {
        frames.clear();
        frames.resize(static_cast<size_t>(numFrames));

        for (int f = 0; f < numFrames; ++f)
        {
            frames[static_cast<size_t>(f)].resize(static_cast<size_t>(tableSize));

            const float progress =
                static_cast<float>(f) / static_cast<float>(numFrames - 1);

            const int maxHarmonic = 1 + static_cast<int>(progress * 23.0f);

            for (int i = 0; i < tableSize; ++i)
            {
                const double phase =
                    static_cast<double>(i) / static_cast<double>(tableSize);

                double value = 0.0;

                for (int h = 1; h <= maxHarmonic; ++h)
                    value += std::sin(juce::MathConstants<double>::twoPi *
                        static_cast<double>(h) * phase) /
                    static_cast<double>(h);

                frames[static_cast<size_t>(f)][static_cast<size_t>(i)] =
                    static_cast<float>(value);
            }

            normalizeFrame(f);
        }
    }

    void normalizeFrame(int index)
    {
        if (index < 0 || index >= static_cast<int>(frames.size()))
            return;

        auto& fr = frames[static_cast<size_t>(index)];

        double mean = 0.0;
        for (float v : fr)
            mean += v;
        mean /= static_cast<double>(fr.size());

        float maxAbs = 0.0f;

        for (auto& v : fr)
        {
            v = static_cast<float>(static_cast<double>(v) - mean);
            maxAbs = juce::jmax(maxAbs, std::abs(v));
        }

        if (maxAbs > 0.0001f)
        {
            const float gain = 1.0f / maxAbs;
            for (auto& v : fr)
                v *= gain;
        }
    }

    float getSample(double phase, float position) const
    {
        if (frames.empty())
            return 0.0f;

        position = juce::jlimit(0.0f, 1.0f, position);

        const float frameFloat = position * static_cast<float>(numFrames - 1);

        int frame0 = juce::jlimit(0, numFrames - 1, static_cast<int>(frameFloat));
        int frame1 = juce::jmin(numFrames - 1, frame0 + 1);

        const float frameMix = frameFloat - static_cast<float>(frame0);

        int sampleIndex = juce::jlimit(0, tableSize - 1,
            static_cast<int>(phase * tableSize));

        const int nextIndex = (sampleIndex + 1) % tableSize;

        const float sampleFrac =
            static_cast<float>(phase * tableSize - sampleIndex);

        const float a0 = frames[static_cast<size_t>(frame0)][static_cast<size_t>(sampleIndex)];
        const float a1 = frames[static_cast<size_t>(frame0)][static_cast<size_t>(nextIndex)];
        const float b0 = frames[static_cast<size_t>(frame1)][static_cast<size_t>(sampleIndex)];
        const float b1 = frames[static_cast<size_t>(frame1)][static_cast<size_t>(nextIndex)];

        const float s0 = a0 + (a1 - a0) * sampleFrac;
        const float s1 = b0 + (b1 - b0) * sampleFrac;

        return s0 + (s1 - s0) * frameMix;
    }
};

//==============================================================================
struct OutputMonitor
{
    static constexpr int scopeSize = 1024;
    static constexpr int scopeMask = scopeSize - 1;

    std::array<std::atomic<float>, scopeSize> scopeL;
    std::array<std::atomic<float>, scopeSize> scopeR;
    std::atomic<int> writePos{ 0 };

    std::atomic<float> peakL{ 0.0f };
    std::atomic<float> peakR{ 0.0f };

    OutputMonitor()
    {
        for (auto& a : scopeL)
            a.store(0.0f, std::memory_order_relaxed);
        for (auto& a : scopeR)
            a.store(0.0f, std::memory_order_relaxed);
    }

    void pushBlock(const float* l, const float* r, int n)
    {
        int pos = writePos.load(std::memory_order_relaxed);

        for (int i = 0; i < n; ++i)
        {
            scopeL[static_cast<size_t>(pos)].store(l[i], std::memory_order_relaxed);
            scopeR[static_cast<size_t>(pos)].store(r[i], std::memory_order_relaxed);
            pos = (pos + 1) & scopeMask;
        }

        writePos.store(pos, std::memory_order_relaxed);
    }

    void pushPeaks(float pl, float pr)
    {
        peakL.store(juce::jmax(peakL.load(std::memory_order_relaxed), pl),
            std::memory_order_relaxed);
        peakR.store(juce::jmax(peakR.load(std::memory_order_relaxed), pr),
            std::memory_order_relaxed);
    }

    float takePeakL() { return peakL.exchange(0.0f, std::memory_order_relaxed); }
    float takePeakR() { return peakR.exchange(0.0f, std::memory_order_relaxed); }

    void copyScope(std::vector<float>& outL, std::vector<float>& outR) const
    {
        const int pos = writePos.load(std::memory_order_relaxed);

        for (int i = 0; i < scopeSize; ++i)
        {
            const int idx = (pos + i) & scopeMask;
            outL[static_cast<size_t>(i)] =
                scopeL[static_cast<size_t>(idx)].load(std::memory_order_relaxed);
            outR[static_cast<size_t>(i)] =
                scopeR[static_cast<size_t>(idx)].load(std::memory_order_relaxed);
        }
    }
};

//==============================================================================
struct SynthParamRefs
{
    std::atomic<float>* cutoff = nullptr;
    std::atomic<float>* resonance = nullptr;
    std::atomic<float>* attack = nullptr;
    std::atomic<float>* decay = nullptr;
    std::atomic<float>* sustain = nullptr;
    std::atomic<float>* release = nullptr;
    std::atomic<float>* detune = nullptr;
    std::atomic<float>* volume = nullptr;

    std::atomic<float>* lfoRate = nullptr;
    std::atomic<float>* lfoDepth = nullptr;
    std::atomic<float>* wtPos = nullptr;

    std::atomic<float>* osc1Shape = nullptr;
    std::atomic<float>* osc2Shape = nullptr;
    std::atomic<float>* osc3Shape = nullptr;
    std::atomic<float>* osc1Level = nullptr;
    std::atomic<float>* osc2Level = nullptr;
    std::atomic<float>* osc3Level = nullptr;
    std::atomic<float>* osc3Semi = nullptr;

    std::atomic<float>* voiceMode = nullptr;
    std::atomic<float>* glide = nullptr;

    std::atomic<float>* uniVoices = nullptr;
    std::atomic<float>* uniDetune = nullptr;
    std::atomic<float>* uniSpread = nullptr;

    std::atomic<float>* fxDrive = nullptr;
    std::atomic<float>* chorusRate = nullptr;
    std::atomic<float>* chorusDepth = nullptr;
    std::atomic<float>* chorusMix = nullptr;
    std::atomic<float>* delayTime = nullptr;
    std::atomic<float>* delayFeedback = nullptr;
    std::atomic<float>* delayMix = nullptr;
    std::atomic<float>* reverbSize = nullptr;
    std::atomic<float>* reverbMix = nullptr;

    std::atomic<SampleData*>* sample = nullptr;

    std::atomic<float>* sampleLevel = nullptr;
    std::atomic<float>* sampleMode = nullptr;
    std::atomic<float>* sampleRoot = nullptr;
    std::atomic<float>* sampleStart = nullptr;
    std::atomic<float>* sampleReverse = nullptr;

    std::atomic<WavetableData*>* wavetable = nullptr;

    std::atomic<float>* filterType = nullptr;
    std::atomic<float>* filterDrive = nullptr;
    std::atomic<float>* filterKeytrack = nullptr;
    std::atomic<float>* fenvAmount = nullptr;
    std::atomic<float>* fenvAttack = nullptr;
    std::atomic<float>* fenvDecay = nullptr;
    std::atomic<float>* fenvSustain = nullptr;
    std::atomic<float>* fenvRelease = nullptr;

    std::atomic<float>* wtWarp = nullptr;
    std::atomic<float>* wtWarpAmt = nullptr;
    
    std::atomic<float>* pwmWidth = nullptr;
    std::atomic<float>* fmAmount = nullptr;
    std::atomic<float>* hardSync = nullptr;

    
    std::atomic<float>* lfo2Rate = nullptr;
    std::atomic<float>* lfo2Depth = nullptr;
    std::atomic<float>* lfo2Shape = nullptr;
    std::atomic<float>* lfo2Sync = nullptr;

    std::atomic<float>* modSrc1 = nullptr;
    std::atomic<float>* modDest1 = nullptr;
    std::atomic<float>* modAmt1 = nullptr;
    std::atomic<float>* modSrc2 = nullptr;
    std::atomic<float>* modDest2 = nullptr;
    std::atomic<float>* modAmt2 = nullptr;
    std::atomic<float>* modSrc3 = nullptr;
    std::atomic<float>* modDest3 = nullptr;
    std::atomic<float>* modAmt3 = nullptr;
    std::atomic<float>* modSrc4 = nullptr;
    std::atomic<float>* modDest4 = nullptr;
    std::atomic<float>* modAmt4 = nullptr;

    std::atomic<float>* lastBpmRef = nullptr;

    std::atomic<float>* modWheel = nullptr;
    std::atomic<float>* aftertouch = nullptr;

    std::atomic<float>* mpeOn = nullptr;
    std::atomic<float>* mpeBend = nullptr;
};

//==============================================================================
class SynthSound : public juce::SynthesiserSound
{
public:
    bool appliesToNote(int) override { return true; }
    bool appliesToChannel(int) override { return true; }
};

//==============================================================================
class SynthVoice : public juce::SynthesiserVoice
{
public:
    explicit SynthVoice(SynthParamRefs refs);

    bool canPlaySound(juce::SynthesiserSound* sound) override;

    void startNote(int midiNoteNumber, float velocity,
        juce::SynthesiserSound* sound,
        int currentPitchWheelPosition) override;

    void stopNote(float velocity, bool allowTailOff) override;

    void pitchWheelMoved(int newValue) override;
    void controllerMoved(int controllerNumber, int newValue) override;

    void setMpeChannel(int ch) { mpeChannel = ch; }
    void setPressure(float v) { notePressure = v; }

    void renderNextBlock(juce::AudioBuffer<float>& outputBuffer,
        int startSample, int numSamples) override;

    void glideToNote(int midiNote, bool retrigger);

private:
    //==========================================================================
    struct Oscillator
    {
        double phase = 0.0;
        double phaseIncrement = 0.0;

        float pulseWidth = 0.5f;
        juce::uint32 rng = 0x9e3779b9u;

        void reset() { phase = 0.0; }

        void setFrequency(double sampleRate, double frequency)
        {
            if (sampleRate > 0.0)
                phaseIncrement = frequency / sampleRate;
        }

        void advance()
        {
            phase += phaseIncrement;
            if (phase >= 1.0) phase -= 1.0;
            if (phase < 0.0) phase += 1.0;
        }

        float next(int shape)
        {
            const float p = static_cast<float>(phase);
            advance();

            switch (shape)
            {
            case 0: return std::sin(juce::MathConstants<float>::twoPi * p);
            case 1: return 2.0f * p - 1.0f;
            case 2: return p < 0.5f ? 1.0f : -1.0f;
            case 3: return 4.0f * std::abs(p - 0.5f) - 1.0f;

            case 5: // PWM
                return p < pulseWidth ? 1.0f : -1.0f;

            case 6: // Noise
            {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                return static_cast<float>(rng & 0xFFFFu) / 32768.0f - 1.0f;
            }

            default: return 0.0f;
            }
        }
    };

    //==========================================================================
    struct Filter
    {
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;

        void reset()
        {
            x1 = x2 = y1 = y2 = 0.0f;
            b0 = 1.0f; b1 = 0.0f; b2 = 0.0f; a1 = 0.0f; a2 = 0.0f;
        }

        void update(int type, double sampleRate, float frequency, float resonance)
        {
            if (sampleRate <= 0.0)
                return;

            const float sr = static_cast<float>(sampleRate);
            const float fc = juce::jlimit(20.0f, sr * 0.45f, frequency);
            const float q = juce::jlimit(0.3f, 12.0f, 0.707f + resonance * 8.0f);

            const float omega = juce::MathConstants<float>::twoPi * fc / sr;
            const float sn = std::sin(omega);
            const float cs = std::cos(omega);
            const float alpha = sn / (2.0f * q);

            float nb0 = 0.0f, nb1 = 0.0f, nb2 = 0.0f;

            switch (type)
            {
            case 1:
                nb0 = (1.0f + cs) * 0.5f; nb1 = -(1.0f + cs); nb2 = nb0;
                break;
            case 2:
                nb0 = alpha; nb1 = 0.0f; nb2 = -alpha;
                break;
            case 3:
                nb0 = 1.0f; nb1 = -2.0f * cs; nb2 = 1.0f;
                break;
            default:
                nb0 = (1.0f - cs) * 0.5f; nb1 = 1.0f - cs; nb2 = nb0;
                break;
            }

            const float a0 = 1.0f + alpha;

            b0 = nb0 / a0; b1 = nb1 / a0; b2 = nb2 / a0;
            a1 = (-2.0f * cs) / a0;
            a2 = (1.0f - alpha) / a0;
        }

        float process(float input)
        {
            float output = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

            if (!std::isfinite(output))
                output = 0.0f;

            x2 = x1; x1 = input;
            y2 = y1; y1 = output;

            return output;
        }
    };

    //==========================================================================
    float renderOsc(Oscillator& osc, int shape, float wtPos);
    static float renderLfoValue(int shape, float phase);
    float nextRandom();

    //==========================================================================
    static constexpr int maxUni = 5;

    SynthParamRefs params;

    Oscillator oscs[3][maxUni];

    Filter filterL;
    Filter filterR;

    WavetableData wavetable;
    const WavetableData* wtActive = nullptr;

    juce::ADSR adsr;
    juce::ADSR fenv;

    float lastFenv = 0.0f;
    int noteNumber = 60;
    float noteVelocity = 0.0f;

    int wtWarpMode = 0;
    float wtWarpAmt = 0.5f;

    int mpeChannel = 1;
    float notePressure = 0.0f;
    float noteTimbre = 0.0f;

    double noteHz = 440.0;
    double targetNoteHz = 440.0;

    double samplePos = 0.0;
    bool samplePlaying = false;

    int pitchWheel = 8192;

    double lfoPhase = 0.0;
    double lfo2Phase = 0.0;
    float lfo2ShValue = 0.0f;

    juce::uint32 rngState = 0x1234567u;
};

//==============================================================================
class StraticSynthesiser : public juce::Synthesiser
{
public:
    SynthParamRefs params;

    void handleMidiEvent(const juce::MidiMessage& m) override;
    void allNotesOff(int midiChannel, bool shouldStopSustainedNotes) override;

private:
    SynthVoice* findActiveVoice();

    std::vector<int> heldNotes;
    int baseNote = -1;
    int baseChannel = 1;
};

//==============================================================================
struct FxParams
{
    float drive = 0.0f;
    float chorusRate = 0.8f;
    float chorusDepth = 0.3f;
    float chorusMix = 0.0f;
    float delayTime = 0.25f;
    float delayFeedback = 0.35f;
    float delayMix = 0.0f;
    float reverbSize = 0.5f;
    float reverbMix = 0.0f;

    float eqLow = 0.0f;
    float eqMid = 0.0f;
    float eqHigh = 0.0f;

    bool compOn = false;
    float compThresh = -12.0f;
    float compRatio = 4.0f;
    float compAttack = 10.0f;
    float compRelease = 200.0f;
};
//==============================================================================
struct FxChain
{
    void prepare(double sampleRate)
    {
        sr = sampleRate;

        const int delaySize = static_cast<int>(sr * 1.0) + 4;
        const int chorusSize = static_cast<int>(sr * 0.04) + 4;

        for (int ch = 0; ch < 2; ++ch)
        {
            delayBuf[ch].assign(static_cast<size_t>(delaySize), 0.0f);
            delayWrite[ch] = 0;
            chorusBuf[ch].assign(static_cast<size_t>(chorusSize), 0.0f);
            chorusWrite[ch] = 0;
        }

        chorusPhase = 0.0;
        reverb.setSampleRate(sampleRate);

        for (int ch = 0; ch < 2; ++ch)
        {
            eqLowB[ch].reset();
            eqMidB[ch].reset();
            eqHighB[ch].reset();
        }

        detEnv = 0.0f;
        compGain = 1.0f;
    }

    void reset()
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            std::fill(delayBuf[ch].begin(), delayBuf[ch].end(), 0.0f);
            std::fill(chorusBuf[ch].begin(), chorusBuf[ch].end(), 0.0f);
        }
    }

    static float readInterp(const std::vector<float>& buf, float pos)
    {
        const int n = static_cast<int>(buf.size());
        if (n <= 0) return 0.0f;

        int i0 = static_cast<int>(std::floor(pos));
        const float frac = pos - static_cast<float>(i0);

        i0 = ((i0 % n) + n) % n;
        const int i1 = (i0 + 1) % n;

        return buf[static_cast<size_t>(i0)] +
            (buf[static_cast<size_t>(i1)] - buf[static_cast<size_t>(i0)]) * frac;
    }

    void process(juce::AudioBuffer<float>& buffer, const FxParams& p)
    {
        const int numSamples = buffer.getNumSamples();

        float* ch0 = buffer.getWritePointer(0);
        float* ch1 = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;

        if (p.drive > 0.001f)
        {
            const float k = 1.0f + p.drive * 30.0f;
            const float norm = 1.0f / std::tanh(k);

            for (int i = 0; i < numSamples; ++i)
            {
                ch0[i] = std::tanh(ch0[i] * k) * norm;
                if (ch1 != nullptr)
                    ch1[i] = std::tanh(ch1[i] * k) * norm;
            }
        }

        if (p.chorusMix > 0.001f && !chorusBuf[0].empty())
        {
            const int n = static_cast<int>(chorusBuf[0].size());
            const float baseDelay = static_cast<float>(n) * 0.35f;
            const float depth = static_cast<float>(n) * 0.25f * p.chorusDepth;
            const float phaseInc = static_cast<float>(p.chorusRate / sr);

            for (int i = 0; i < numSamples; ++i)
            {
                chorusPhase += phaseInc;
                if (chorusPhase >= 1.0f) chorusPhase -= 1.0f;

                const float lfo = std::sin(juce::MathConstants<float>::twoPi * chorusPhase);
                const float delaySamples = baseDelay + lfo * depth;

                for (int ch = 0; ch < 2; ++ch)
                {
                    float* dst = (ch == 0) ? ch0 : ch1;
                    if (dst == nullptr) continue;

                    chorusBuf[ch][static_cast<size_t>(chorusWrite[ch])] = dst[i];

                    float readPos = static_cast<float>(chorusWrite[ch]) - delaySamples;
                    if (readPos < 0.0f) readPos += static_cast<float>(n);

                    const float wet = readInterp(chorusBuf[ch], readPos);

                    dst[i] = dst[i] * (1.0f - p.chorusMix) + wet * p.chorusMix;

                    chorusWrite[ch] = (chorusWrite[ch] + 1) % n;
                }
            }
        }

        if (!delayBuf[0].empty())
        {
            const int n = static_cast<int>(delayBuf[0].size());

            const float delaySamples = juce::jlimit(
                1.0f, static_cast<float>(n - 2),
                static_cast<float>(p.delayTime * sr));

            const bool active = p.delayMix > 0.001f;
            const float fb = juce::jlimit(0.0f, 0.9f, p.delayFeedback);

            for (int i = 0; i < numSamples; ++i)
            {
                for (int ch = 0; ch < 2; ++ch)
                {
                    float* dst = (ch == 0) ? ch0 : ch1;
                    if (dst == nullptr) continue;

                    float readPos = static_cast<float>(delayWrite[ch]) - delaySamples;
                    if (readPos < 0.0f) readPos += static_cast<float>(n);

                    const float delayed = readInterp(delayBuf[ch], readPos);

                    delayBuf[ch][static_cast<size_t>(delayWrite[ch])] =
                        dst[i] + delayed * fb;

                    delayWrite[ch] = (delayWrite[ch] + 1) % n;

                    if (active)
                        dst[i] = dst[i] + delayed * p.delayMix;
                }
            }
        }

        if (p.reverbMix > 0.001f)
        {
            juce::Reverb::Parameters rp;
            rp.roomSize = juce::jlimit(0.0f, 1.0f, p.reverbSize);
            rp.damping = 0.5f;
            rp.wetLevel = p.reverbMix;
            rp.dryLevel = 1.0f;
            rp.width = 1.0f;

            reverb.setParameters(rp);

            if (ch1 != nullptr)
                reverb.processStereo(ch0, ch1, numSamples);
            else
                reverb.processMono(ch0, numSamples);
        }

        //======================================================================
        // EQ
        for (int ch = 0; ch < 2; ++ch)
        {
            eqLowB[ch].setLowShelf(sr, 200.0f, p.eqLow);
            eqMidB[ch].setPeak(sr, 1000.0f, p.eqMid, 1.0f);
            eqHighB[ch].setHighShelf(sr, 4000.0f, p.eqHigh);
        }

        for (int ch = 0; ch < 2; ++ch)
        {
            float* dst = (ch == 0) ? ch0 : ch1;

            if (dst == nullptr)
                continue;

            for (int i = 0; i < numSamples; ++i)
                dst[i] = eqHighB[ch].process(eqMidB[ch].process(eqLowB[ch].process(dst[i])));
        }

        //======================================================================
        // Compressor
        if (p.compOn)
        {
            const float threshLin = std::pow(10.0f, p.compThresh / 20.0f);
            const float attC = std::exp(-1.0f / (p.compAttack * 0.001f * static_cast<float>(sr) + 0.0001f));
            const float relC = std::exp(-1.0f / (p.compRelease * 0.001f * static_cast<float>(sr) + 0.0001f));

            for (int i = 0; i < numSamples; ++i)
            {
                const float peak = juce::jmax(std::abs(ch0[i]),
                    ch1 != nullptr ? std::abs(ch1[i]) : 0.0f);

                if (peak > detEnv)
                    detEnv = peak + attC * (detEnv - peak);
                else
                    detEnv = peak + relC * (detEnv - peak);

                float desired = 1.0f;

                if (detEnv > threshLin)
                {
                    const float overDb = juce::Decibels::gainToDecibels(detEnv / threshLin);
                    const float redDb = overDb * (1.0f / p.compRatio - 1.0f);
                    desired = std::pow(10.0f, redDb / 20.0f);
                }

                if (desired < compGain)
                    compGain = desired + attC * (compGain - desired);
                else
                    compGain = desired + relC * (compGain - desired);

                ch0[i] *= compGain;

                if (ch1 != nullptr)
                    ch1[i] *= compGain;
            }
        }

        //======================================================================
        // brickwall limiter
        for (int i = 0; i < numSamples; ++i)
        {
            ch0[i] = juce::jlimit(-1.0f, 1.0f, ch0[i]);

            if (ch1 != nullptr)
                ch1[i] = juce::jlimit(-1.0f, 1.0f, ch1[i]);
        }
    }

    double sr = 44100.0;

    std::vector<float> delayBuf[2];
    std::vector<float> chorusBuf[2];
    int delayWrite[2] = { 0, 0 };
    int chorusWrite[2] = { 0, 0 };
    float chorusPhase = 0.0f;

    juce::Reverb reverb;

    //==========================================================================
    struct Biquad
    {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;

        void reset() { x1 = x2 = y1 = y2 = 0.0f; }

        void setLowShelf(double sr, float freq, float gainDb)
        {
            const float A = std::pow(10.0f, gainDb / 40.0f);
            const float w = juce::MathConstants<float>::twoPi * freq / static_cast<float>(sr);
            const float cs = std::cos(w);
            const float sn = std::sin(w);
            const float alpha = sn / (2.0f * 0.707f);
            const float beta = 2.0f * std::sqrt(A) * alpha;

            const float a0 = (A + 1.0f) - (A - 1.0f) * cs + beta;
            b0 = A * ((A + 1.0f) + (A - 1.0f) * cs + beta) / a0;
            b1 = 2.0f * A * ((A - 1.0f) + (A + 1.0f) * cs) / a0;
            b2 = A * ((A + 1.0f) - (A - 1.0f) * cs - beta) / a0;
            a1 = -2.0f * ((A - 1.0f) - (A + 1.0f) * cs) / a0;
            a2 = ((A + 1.0f) - (A - 1.0f) * cs - beta) / a0;
        }

        void setHighShelf(double sr, float freq, float gainDb)
        {
            const float A = std::pow(10.0f, gainDb / 40.0f);
            const float w = juce::MathConstants<float>::twoPi * freq / static_cast<float>(sr);
            const float cs = std::cos(w);
            const float sn = std::sin(w);
            const float alpha = sn / (2.0f * 0.707f);
            const float beta = 2.0f * std::sqrt(A) * alpha;

            const float a0 = (A + 1.0f) + (A - 1.0f) * cs + beta;
            b0 = A * ((A + 1.0f) - (A - 1.0f) * cs + beta) / a0;
            b1 = -2.0f * A * ((A - 1.0f) - (A + 1.0f) * cs) / a0;
            b2 = A * ((A + 1.0f) - (A - 1.0f) * cs - beta) / a0;
            a1 = 2.0f * ((A - 1.0f) + (A + 1.0f) * cs) / a0;
            a2 = ((A + 1.0f) + (A - 1.0f) * cs - beta) / a0;
        }

        void setPeak(double sr, float freq, float gainDb, float q)
        {
            const float A = std::pow(10.0f, gainDb / 40.0f);
            const float w = juce::MathConstants<float>::twoPi * freq / static_cast<float>(sr);
            const float cs = std::cos(w);
            const float sn = std::sin(w);
            const float alpha = sn / (2.0f * q);

            const float a0 = 1.0f + alpha / A;
            b0 = (1.0f + alpha * A) / a0;
            b1 = (-2.0f * cs) / a0;
            b2 = (1.0f - alpha * A) / a0;
            a1 = (-2.0f * cs) / a0;
            a2 = (1.0f - alpha / A) / a0;
        }

        float process(float in)
        {
            const float out = b0 * in + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

            x2 = x1; x1 = in;
            y2 = y1; y1 = out;

            return std::isfinite(out) ? out : 0.0f;
        }
    };

    Biquad eqLowB[2];
    Biquad eqMidB[2];
    Biquad eqHighB[2];

    float detEnv = 0.0f;
    float compGain = 1.0f;
};

//==============================================================================
class StraticSynthAudioProcessor : public juce::AudioProcessor
{
public:
    juce::AudioProcessorValueTreeState parameters;

    struct Preset
    {
        juce::String name;
        juce::ValueTree state;
    };

    StraticSynthAudioProcessor();
    ~StraticSynthAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    void processBlock(juce::AudioBuffer<float>& buffer,
        juce::MidiBuffer& midiMessages) override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    int getNumPresets() const;
    juce::String getPresetName(int index) const;
    int getCurrentPresetIndex() const;
    juce::String getCurrentPresetName() const;

    void loadPresetByIndex(int index);
    void nextPreset();
    void prevPreset();

    bool savePresetToFile(const juce::File& file, const juce::String& name);
    bool loadPresetFromFile(const juce::File& file);

    void saveCurrentAsUserPreset(const juce::String& name);

    bool exportBankToFile(const juce::File& file);
    bool importBankFromFile(const juce::File& file);

    void scanUserPresets();
    juce::File getUserPresetDirectory() const;

    bool loadSampleFromFile(const juce::File& file);
    juce::String getSampleName() const;

    bool loadWavetableFromFile(const juce::File& file);
    void resetWavetable();

    void applyWavetable(const WavetableData& data);

    std::atomic<WavetableData*>* getWavetableSource() { return &currentWavetable; }
    OutputMonitor& getMonitor() { return monitor; }
    std::atomic<float>* getModWheelSource() { return &modWheelValue; }
    std::atomic<float>* getAftertouchSource() { return &aftertouchValue; }

    void startMidiLearn(const juce::String& paramID);
    void cancelMidiLearn();
    bool isMidiLearnPending() const;
    int getMappingCC(const juce::String& paramID) const;
    void clearMapping(const juce::String& paramID);
    std::vector<std::pair<juce::String, int>> getMappings() const;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    void buildFactoryPresets();
    std::vector<int> buildArpSeq(int mode, int octaves) const;
    void applyPresetValues(std::initializer_list<std::pair<const char*, double>> values);

    std::vector<Preset> factoryPresets;
    std::vector<Preset> userPresets;
    int currentPresetIndex = 0;

    StraticSynthesiser synth;

    FxChain fxChain;
    juce::AudioBuffer<float> fxWork;

    OutputMonitor monitor;
    std::atomic<float> lastBpm{ 120.0f };

    std::atomic<float> modWheelValue{ 0.0f };
    std::atomic<float> aftertouchValue{ 0.0f };

    struct MidiMapping
    {
        juce::String paramID;
        int cc = -1;
    };

    std::vector<MidiMapping> midiMappings;
    juce::String pendingLearnParam;
    juce::CriticalSection midiMapLock;

    
    std::vector<int> heldArp;
    int arpStep = 0;
    double nextStepAbs = 0.0;
    double pendingOffAbs = 0.0;
    int lastArpNote = -1;
    int lastArpVel = 100;
    double absoluteSamplePos = 0.0;

    std::vector<std::unique_ptr<SampleData>> sampleHistory;
    std::atomic<SampleData*> currentSample{ nullptr };

    std::vector<std::unique_ptr<WavetableData>> wtHistory;
    std::atomic<WavetableData*> currentWavetable{ nullptr };

    juce::AudioFormatManager formatManager;
    juce::String sampleName;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StraticSynthAudioProcessor)
};