#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

//==============================================================================
SynthVoice::SynthVoice(SynthParamRefs refs)
    : params(refs)
{
}

bool SynthVoice::canPlaySound(juce::SynthesiserSound* sound)
{
    return dynamic_cast<SynthSound*>(sound) != nullptr;
}

//==============================================================================
void SynthVoice::startNote(int midiNoteNumber, float velocity,
    juce::SynthesiserSound*, int currentPitchWheelPosition)
{
    noteVelocity = velocity;
    noteHz = juce::MidiMessage::getMidiNoteInHertz(midiNoteNumber);
    targetNoteHz = noteHz;
    pitchWheel = currentPitchWheelPosition;
    noteNumber = midiNoteNumber;

    for (int o = 0; o < 3; ++o)
    {
        for (int u = 0; u < maxUni; ++u)
        {
            oscs[o][u].reset();
            const double golden = 0.6180339887498949 * static_cast<double>(u);
            oscs[o][u].phase = golden - std::floor(golden);
        }
    }

    filterL.reset();
    filterR.reset();

    notePressure = 0.0f;
    noteTimbre = 0.0f;
    mpeChannel = 1;

    {
        SampleData* smp = params.sample != nullptr ? params.sample->load() : nullptr;
        const int len = smp != nullptr ? smp->buffer.getNumSamples() : 0;

        if (len > 1)
        {
            const float startFrac = params.sampleStart != nullptr
                ? params.sampleStart->load() : 0.0f;
            const bool rev = params.sampleReverse != nullptr &&
                static_cast<int>(params.sampleReverse->load()) == 1;

            const double last = static_cast<double>(len - 1);

            samplePos = rev ? last * (1.0 - static_cast<double>(startFrac))
                : last * static_cast<double>(startFrac);
            samplePlaying = true;
        }
        else
        {
            samplePos = 0.0;
            samplePlaying = false;
        }
    }

    adsr.setSampleRate(static_cast<float>(getSampleRate()));
    adsr.setParameters({
        params.attack->load(), params.decay->load(),
        params.sustain->load(), params.release->load() });
    adsr.noteOn();

    fenv.setSampleRate(static_cast<float>(getSampleRate()));
    fenv.setParameters({
        params.fenvAttack != nullptr ? params.fenvAttack->load() : 0.005f,
        params.fenvDecay != nullptr ? params.fenvDecay->load() : 0.3f,
        params.fenvSustain != nullptr ? params.fenvSustain->load() : 0.5f,
        params.fenvRelease != nullptr ? params.fenvRelease->load() : 0.3f });
    fenv.noteOn();
}

float noteVelocityGlobal = 0.0f; // not used; placeholder removed below

//==============================================================================
void SynthVoice::stopNote(float, bool allowTailOff)
{
    if (allowTailOff)
    {
        adsr.noteOff();
        fenv.noteOff();
    }
    else
    {
        adsr.reset();
        fenv.reset();
        clearCurrentNote();
    }
}

void SynthVoice::pitchWheelMoved(int newValue) { pitchWheel = newValue; }

void SynthVoice::controllerMoved(int controllerNumber, int newValue)
{
    if (controllerNumber == 74)
        noteTimbre = static_cast<float>(newValue) / 127.0f;
}


void SynthVoice::glideToNote(int midiNote, bool retrigger)
{
    targetNoteHz = juce::MidiMessage::getMidiNoteInHertz(midiNote);

    if (retrigger)
    {
        adsr.reset();
        adsr.noteOn();
    }
}

//==============================================================================
float SynthVoice::nextRandom()
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return (static_cast<float>(rngState & 0xFFFFu) / 32768.0f) - 1.0f;
}

float SynthVoice::renderLfoValue(int shape, float phase)
{
    switch (shape)
    {
    case 0: return std::sin(juce::MathConstants<float>::twoPi * phase);
    case 1: return 4.0f * std::abs(phase - 0.5f) * 2.0f - 1.0f;
    case 2: return 2.0f * phase - 1.0f;
    case 3: return phase < 0.5f ? 1.0f : -1.0f;
    default: return std::sin(juce::MathConstants<float>::twoPi * phase);
    }
}

float SynthVoice::renderOsc(Oscillator& osc, int shape, float wtPos)
{
    if (shape == 4)
    {
        const WavetableData& table =
            (wtActive != nullptr && !wtActive->frames.empty()) ? *wtActive : wavetable;

        double ph = osc.phase;

        //======================================================================
        // Bend: нелинейный варп фазы
        if (wtWarpMode == 1)
        {
            const double k = static_cast<double>(wtWarpAmt) * 0.45;

            double warped = ph + k * std::sin(juce::MathConstants<double>::twoPi * ph) /
                juce::MathConstants<double>::twoPi;

            warped -= std::floor(warped);
            ph = warped;
        }

        float s = table.getSample(ph, wtPos);

        //======================================================================
        if (wtWarpMode == 2) // Asym
        {
            const float k = wtWarpAmt;
            s = s + k * (s * s - 1.0f) * 0.5f;
        }
        else if (wtWarpMode == 3) // Quantize
        {
            const float steps = juce::jmax(2.0f, std::round(32.0f - wtWarpAmt * 30.0f));
            s = std::round(s * steps) / steps;
        }
        else if (wtWarpMode == 4) // Smooth
        {
            const double w = 0.002 + static_cast<double>(wtWarpAmt) * 0.05;

            double pa = ph - w;
            if (pa < 0.0) pa += 1.0;

            double pb = ph + w;
            if (pb >= 1.0) pb -= 1.0;

            const float a = table.getSample(pa, wtPos);
            const float b = table.getSample(pb, wtPos);

            s = (a + s + b) / 3.0f;
        }

        osc.advance();
        return s;
    }

    return osc.next(shape);
}

//==============================================================================
void SynthVoice::renderNextBlock(juce::AudioBuffer<float>& outputBuffer,
    int startSample, int numSamples)
{
    if (!adsr.isActive())
    {
        clearCurrentNote();
        return;
    }

    const double sr = getSampleRate();
    if (sr <= 0.0)
        return;

    adsr.setSampleRate(static_cast<float>(sr));
    adsr.setParameters({
        params.attack->load(), params.decay->load(),
        params.sustain->load(), params.release->load() });

    fenv.setSampleRate(static_cast<float>(sr));
    fenv.setParameters({
        params.fenvAttack != nullptr ? params.fenvAttack->load() : 0.005f,
        params.fenvDecay != nullptr ? params.fenvDecay->load() : 0.3f,
        params.fenvSustain != nullptr ? params.fenvSustain->load() : 0.5f,
        params.fenvRelease != nullptr ? params.fenvRelease->load() : 0.3f });

    const float baseCutoff = juce::jlimit(20.0f, static_cast<float>(sr) * 0.45f,
        params.cutoff->load());
    const float resonanceParam = params.resonance->load();

    int filterType = 0;
    if (params.filterType != nullptr)
        filterType = static_cast<int>(params.filterType->load());
    filterType = juce::jlimit(0, 3, filterType);

    const float filterDrive = params.filterDrive != nullptr ? params.filterDrive->load() : 0.0f;
    const float keytrack = params.filterKeytrack != nullptr ? params.filterKeytrack->load() : 0.0f;
    const float fenvAmount = params.fenvAmount != nullptr ? params.fenvAmount->load() : 0.0f;

    int shape1 = 1, shape2 = 1, shape3 = 0;
    if (params.osc1Shape != nullptr) shape1 = static_cast<int>(params.osc1Shape->load());
    if (params.osc2Shape != nullptr) shape2 = static_cast<int>(params.osc2Shape->load());
    if (params.osc3Shape != nullptr) shape3 = static_cast<int>(params.osc3Shape->load());
    shape1 = juce::jlimit(0, 6, shape1);
    shape2 = juce::jlimit(0, 6, shape2);
    shape3 = juce::jlimit(0, 6, shape3);

    float lvl1 = 0.5f, lvl2 = 0.5f, lvl3 = 0.0f;
    if (params.osc1Level != nullptr) lvl1 = params.osc1Level->load();
    if (params.osc2Level != nullptr) lvl2 = params.osc2Level->load();
    if (params.osc3Level != nullptr) lvl3 = params.osc3Level->load();

    const float levelSum = lvl1 + lvl2 + lvl3;
    const float levelNorm = levelSum > 1.0f ? (1.0f / levelSum) : 1.0f;

    const float detuneCents = params.detune != nullptr ? params.detune->load() : 0.0f;
    const float semi3 = params.osc3Semi != nullptr ? params.osc3Semi->load() : -12.0f;

    int uniCount = 1;
    if (params.uniVoices != nullptr)
        uniCount = static_cast<int>(params.uniVoices->load());
    uniCount = juce::jlimit(1, maxUni, uniCount);

    const float uniDet = params.uniDetune != nullptr ? params.uniDetune->load() : 0.0f;
    const float spread = params.uniSpread != nullptr ? params.uniSpread->load() : 0.5f;

    const float lfoRate = params.lfoRate != nullptr ? params.lfoRate->load() : 0.0f;
    const float lfoDepth = params.lfoDepth != nullptr ? params.lfoDepth->load() : 0.0f;
    const float wtPos = params.wtPos != nullptr ? params.wtPos->load() : 0.5f;

    const float lfo2Rate = params.lfo2Rate != nullptr ? params.lfo2Rate->load() : 2.0f;
    const float lfo2Depth = params.lfo2Depth != nullptr ? params.lfo2Depth->load() : 0.0f;

    int lfo2Shape = 0;
    if (params.lfo2Shape != nullptr)
        lfo2Shape = static_cast<int>(params.lfo2Shape->load());
    lfo2Shape = juce::jlimit(0, 4, lfo2Shape);

    int lfo2Sync = 0;
    if (params.lfo2Sync != nullptr)
        lfo2Sync = static_cast<int>(params.lfo2Sync->load());
    lfo2Sync = juce::jlimit(0, 7, lfo2Sync);

    wtWarpMode = params.wtWarp != nullptr
        ? static_cast<int>(params.wtWarp->load()) : 0;
    wtWarpMode = juce::jlimit(0, 4, wtWarpMode);

    wtWarpAmt = params.wtWarpAmt != nullptr ? params.wtWarpAmt->load() : 0.5f;

    const float pwmWidth = params.pwmWidth != nullptr ? params.pwmWidth->load() : 0.5f;
    const float fmAmount = params.fmAmount != nullptr ? params.fmAmount->load() : 0.0f;
    const bool syncOn = params.hardSync != nullptr &&
        static_cast<int>(params.hardSync->load()) == 1;

    for (int o = 0; o < 3; ++o)
        for (int u = 0; u < maxUni; ++u)
            oscs[o][u].pulseWidth = pwmWidth;

    const float glideTime = params.glide != nullptr ? params.glide->load() : 0.0f;

    if (glideTime <= 0.001f)
        noteHz = targetNoteHz;
    else
    {
        const double coeff =
            1.0 - std::exp(-static_cast<double>(numSamples) / (glideTime * sr));
        noteHz += (targetNoteHz - noteHz) * coeff;
    }

    const bool mpeOn = params.mpeOn != nullptr &&
        static_cast<int>(params.mpeOn->load()) == 1;

    const double bendRange = (mpeOn && mpeChannel != 1)
        ? static_cast<double>(params.mpeBend != nullptr
            ? params.mpeBend->load()
            : 48.0f)
        : 2.0;

    const double bend = std::pow(2.0,
        ((static_cast<double>(pitchWheel) - 8192.0) / 8192.0) * (bendRange / 12.0));
    const double detuneRatio = std::pow(2.0, static_cast<double>(detuneCents) / 1200.0);
    const double semiRatio = std::pow(2.0, static_cast<double>(semi3) / 12.0);

    const double freq1 = noteHz * bend;

    float layerRatio[maxUni];
    float layerPanL[maxUni];
    float layerPanR[maxUni];

    for (int u = 0; u < uniCount; ++u)
    {
        const float off = (uniCount == 1)
            ? 0.0f
            : (2.0f * static_cast<float>(u) / static_cast<float>(uniCount - 1) - 1.0f);

        layerRatio[u] = static_cast<float>(
            std::pow(2.0, (static_cast<double>(uniDet) * off) / 1200.0));

        const float t = (off * spread + 1.0f) * 0.5f;
        const float ang = t * 0.5f * juce::MathConstants<float>::pi;

        layerPanL[u] = std::cos(ang) * 1.41421356f;
        layerPanR[u] = std::sin(ang) * 1.41421356f;
    }

    const float uniNorm = 1.0f / std::sqrt(static_cast<float>(uniCount));
    const float ampBase = noteVelocity * 0.35f * uniNorm;

    SampleData* smp = params.sample != nullptr ? params.sample->load() : nullptr;
    wtActive = params.wavetable != nullptr ? params.wavetable->load() : nullptr;

    const float sampleLevel = params.sampleLevel != nullptr ? params.sampleLevel->load() : 0.0f;
    const int sampleLoopMode = params.sampleMode != nullptr
        ? static_cast<int>(params.sampleMode->load()) : 0;
    const bool sampleRev = params.sampleReverse != nullptr &&
        static_cast<int>(params.sampleReverse->load()) == 1;

    int rootNote = 60;
    if (params.sampleRoot != nullptr)
        rootNote = static_cast<int>(params.sampleRoot->load());
    rootNote = juce::jlimit(24, 96, rootNote);
    const double rootHz = juce::MidiMessage::getMidiNoteInHertz(rootNote);

    const double sampleInc = (smp != nullptr && smp->sampleRate > 0.0)
        ? (freq1 / rootHz) * (smp->sampleRate / sr)
        : 0.0;

    const float modAmt[4] = {
        params.modAmt1 != nullptr ? params.modAmt1->load() : 0.0f,
        params.modAmt2 != nullptr ? params.modAmt2->load() : 0.0f,
        params.modAmt3 != nullptr ? params.modAmt3->load() : 0.0f,
        params.modAmt4 != nullptr ? params.modAmt4->load() : 0.0f };

    const int modSrc[4] = {
        params.modSrc1 != nullptr ? static_cast<int>(params.modSrc1->load()) : 0,
        params.modSrc2 != nullptr ? static_cast<int>(params.modSrc2->load()) : 0,
        params.modSrc3 != nullptr ? static_cast<int>(params.modSrc3->load()) : 0,
        params.modSrc4 != nullptr ? static_cast<int>(params.modSrc4->load()) : 0 };

    const int modDest[4] = {
        params.modDest1 != nullptr ? static_cast<int>(params.modDest1->load()) : 0,
        params.modDest2 != nullptr ? static_cast<int>(params.modDest2->load()) : 0,
        params.modDest3 != nullptr ? static_cast<int>(params.modDest3->load()) : 0,
        params.modDest4 != nullptr ? static_cast<int>(params.modDest4->load()) : 0 };

    const float bpm = params.lastBpmRef != nullptr ? params.lastBpmRef->load() : 120.0f;

    int controlCounter = 0;
    const int numCh = outputBuffer.getNumChannels();

    for (int i = 0; i < numSamples; ++i)
    {
        const float env = adsr.getNextSample();
        lastFenv = fenv.getNextSample();

        if (env == 0.0f && !adsr.isActive())
        {
            clearCurrentNote();
            break;
        }

        lfoPhase += static_cast<double>(lfoRate) / sr;
        if (lfoPhase >= 1.0) lfoPhase -= 1.0;

        double lfo2Hz = static_cast<double>(lfo2Rate);

        if (lfo2Sync > 0)
        {
            const float beat = 60.0f / juce::jmax(1.0f, bpm);
            float mul = 1.0f;

            switch (lfo2Sync)
            {
            case 1: mul = 1.0f; break;
            case 2: mul = 1.5f; break;
            case 3: mul = 2.0f; break;
            case 4: mul = 3.0f; break;
            case 5: mul = 4.0f; break;
            case 6: mul = 6.0f; break;
            case 7: mul = 8.0f; break;
            }

            lfo2Hz = static_cast<double>(mul / beat);
        }

        lfo2Phase += lfo2Hz / sr;

        if (lfo2Phase >= 1.0)
        {
            lfo2Phase -= 1.0;
            lfo2ShValue = nextRandom();
        }

        const float lfoValue = std::sin(
            juce::MathConstants<float>::twoPi * static_cast<float>(lfoPhase));

        const float lfo2Raw = (lfo2Shape == 4)
            ? lfo2ShValue
            : renderLfoValue(lfo2Shape,
                static_cast<float>(lfo2Phase));

        const float lfo2Value = lfo2Raw * lfo2Depth;

        const float keytrackNorm = juce::jlimit(-1.0f, 1.0f,
            static_cast<float>(noteNumber - 60) / 60.0f);

        const float modWheelVal = params.modWheel != nullptr
            ? params.modWheel->load() : 0.0f;
        const float aftertouchVal = params.aftertouch != nullptr
            ? params.aftertouch->load() : 0.0f;
        const float modValues[12] = {
            0.0f, lfoValue, lfo2Value, env, lastFenv,
            modWheelVal, noteVelocity, keytrackNorm, nextRandom() * 0.15f,
            aftertouchVal, noteTimbre, notePressure };

        float modWtAdd = 0.0f, modCutMul = 1.0f, modResAdd = 0.0f;
        float modL1 = 0.0f, modL2 = 0.0f, modL3 = 0.0f;
        float modP1 = 1.0f, modP2 = 1.0f, modP3 = 1.0f;
        float modVol = 1.0f, modDrv = 0.0f;

        for (int m = 0; m < 4; ++m)
        {
            const int src = modSrc[m];
            const int dest = modDest[m];

            if (src <= 0 || dest <= 0)
                continue;

            const float mod = modAmt[m] * modValues[src];

            switch (dest)
            {
            case 1:  modWtAdd += mod; break;
            case 2:  modCutMul *= std::pow(2.0f, mod * 4.0f); break;
            case 3:  modResAdd += mod; break;
            case 4:  modL1 += mod; break;
            case 5:  modL2 += mod; break;
            case 6:  modL3 += mod; break;
            case 7:  modP1 *= std::pow(2.0f, mod); break;
            case 8:  modP2 *= std::pow(2.0f, mod); break;
            case 9:  modP3 *= std::pow(2.0f, mod); break;
            case 10: modVol *= (1.0f + mod); break;
            case 11: modDrv += mod; break;
            default: break;
            }
        }

        if (controlCounter == 0)
        {
            float freq = baseCutoff;

            if (keytrack > 0.001f)
                freq *= std::pow(2.0f, keytrack *
                    static_cast<float>(noteNumber - 60) / 12.0f);

            if (fenvAmount != 0.0f && lastFenv > 0.0001f)
                freq *= std::pow(2.0f, fenvAmount * lastFenv * 5.0f);

            freq *= (1.0f + lfoDepth * lfoValue);
            freq *= modCutMul;

            freq = juce::jlimit(20.0f, static_cast<float>(sr) * 0.45f, freq);

            const float res = juce::jlimit(0.0f, 1.0f, resonanceParam + modResAdd);

            filterL.update(filterType, sr, freq, res);
            filterR.update(filterType, sr, freq, res);
        }

        controlCounter = (controlCounter + 1) & 31;

        const float wt = juce::jlimit(0.0f, 1.0f, wtPos + modWtAdd);
        const float l1 = juce::jlimit(0.0f, 1.0f, lvl1 + modL1);
        const float l2 = juce::jlimit(0.0f, 1.0f, lvl2 + modL2);
        const float l3 = juce::jlimit(0.0f, 1.0f, lvl3 + modL3);

        float l = 0.0f, r = 0.0f;

        for (int u = 0; u < uniCount; ++u)
        {
            oscs[0][u].setFrequency(sr, freq1 * layerRatio[u] * modP1);
            oscs[1][u].setFrequency(sr, freq1 * layerRatio[u] * detuneRatio * modP2);
            oscs[2][u].setFrequency(sr, freq1 * layerRatio[u] * semiRatio * modP3);

            // osc3 рендерим первым: он источник FM
            const float o3 = renderOsc(oscs[2][u], shape3, wt);

            if (fmAmount > 0.001f)
            {
                const float fmMod = 1.0f + fmAmount * 8.0f * o3;
                oscs[0][u].phaseIncrement *= fmMod;
                oscs[1][u].phaseIncrement *= fmMod;
            }

            const float phaseBefore = static_cast<float>(oscs[0][u].phase);
            const float o1 = renderOsc(oscs[0][u], shape1, wt);
            const bool wrapped = oscs[0][u].phase < phaseBefore;

            if (syncOn && wrapped)
            {
                oscs[1][u].phase = oscs[0][u].phase;
                oscs[2][u].phase = oscs[0][u].phase;
            }

            const float o2 = renderOsc(oscs[1][u], shape2, wt);

            float s = o1 * l1 + o2 * l2 + o3 * l3;

            s *= levelNorm;

            l += s * layerPanL[u];
            r += s * layerPanR[u];
        }

        if (smp != nullptr && sampleLevel > 0.001f && samplePlaying)
        {
            const int len = smp->buffer.getNumSamples();

            if (len > 1)
            {
                if (samplePos < 0.0 || samplePos >= static_cast<double>(len))
                {
                    if (sampleLoopMode == 1)
                    {
                        double w = std::fmod(samplePos, static_cast<double>(len));
                        if (w < 0.0) w += static_cast<double>(len);
                        samplePos = w;
                    }
                    else
                    {
                        samplePlaying = false;
                    }
                }

                if (samplePlaying)
                {
                    int i0 = juce::jlimit(0, len - 1, static_cast<int>(samplePos));
                    const float frac = static_cast<float>(
                        samplePos - static_cast<double>(i0));
                    const int i1 = juce::jmin(len - 1, i0 + 1);

                    const float* c0 = smp->buffer.getReadPointer(0);
                    const float* c1 = smp->buffer.getNumChannels() > 1
                        ? smp->buffer.getReadPointer(1) : nullptr;

                    const float s0 = c0[i0] + (c0[i1] - c0[i0]) * frac;
                    const float s1 = (c1 != nullptr)
                        ? c1[i0] + (c1[i1] - c1[i0]) * frac : s0;

                    l += s0 * sampleLevel;
                    r += s1 * sampleLevel;

                    samplePos += sampleRev ? -sampleInc : sampleInc;
                }
            }
        }

        const float amp = ampBase * env * modVol;

        l *= amp;
        r *= amp;

        const float driveNow = juce::jlimit(0.0f, 1.0f, filterDrive + modDrv);

        if (driveNow > 0.001f)
        {
            const float k = 1.0f + driveNow * 8.0f;
            const float comp = 1.0f / (1.0f + driveNow * 2.0f);

            l = std::tanh(l * k) * comp;
            r = std::tanh(r * k) * comp;
        }

        l = filterL.process(l);
        r = filterR.process(r);

        if (!std::isfinite(l)) l = 0.0f;
        if (!std::isfinite(r)) r = 0.0f;

        if (numCh >= 2)
        {
            outputBuffer.addSample(0, startSample + i, l);
            outputBuffer.addSample(1, startSample + i, r);
        }
        else if (numCh == 1)
        {
            outputBuffer.addSample(0, startSample + i, (l + r) * 0.5f);
        }
    }
}

//==============================================================================
SynthVoice* StraticSynthesiser::findActiveVoice()
{
    for (int i = 0; i < getNumVoices(); ++i)
    {
        if (auto* v = dynamic_cast<SynthVoice*>(getVoice(i)))
            if (v->isVoiceActive())
                return v;
    }

    return nullptr;
}

void StraticSynthesiser::handleMidiEvent(const juce::MidiMessage& m)
{
    const bool mpe = params.mpeOn != nullptr &&
        static_cast<int>(params.mpeOn->load()) == 1;

    int mode = params.voiceMode != nullptr
        ? static_cast<int>(params.voiceMode->load()) : 0;

    if (mpe)
        mode = 0;

    //==========================================================================
    if (m.isNoteOn())
    {
        heldNotes.push_back(m.getNoteNumber());

        if (mode != 0)
        {
            if (SynthVoice* v = findActiveVoice())
            {
                v->glideToNote(m.getNoteNumber(), mode == 1);
                return;
            }
        }

        baseChannel = m.getChannel();
        baseNote = m.getNoteNumber();

        juce::Synthesiser::handleMidiEvent(m);

        if (mpe)
        {
            for (int i = 0; i < getNumVoices(); ++i)
            {
                if (auto* v = dynamic_cast<SynthVoice*>(getVoice(i)))
                {
                    if (v->isVoiceActive() &&
                        v->isPlayingChannel(m.getChannel()) &&
                        v->getCurrentlyPlayingNote() == m.getNoteNumber())
                        v->setMpeChannel(m.getChannel());
                }
            }
        }

        return;
    }

    //==========================================================================
    if (m.isNoteOff())
    {
        for (int i = static_cast<int>(heldNotes.size()) - 1; i >= 0; --i)
        {
            if (heldNotes[static_cast<size_t>(i)] == m.getNoteNumber())
            {
                heldNotes.erase(heldNotes.begin() + i);
                break;
            }
        }

        // моно/легато: есть зажатые клавиши -> переезд на последнюю
        if (mode != 0 && !heldNotes.empty())
        {
            if (SynthVoice* v = findActiveVoice())
            {
                v->glideToNote(heldNotes.back(), false);
                return;
            }
        }

        // моно/легато: клавиш не осталось -> всегда гасим голос напрямую
        if (mode != 0)
        {
            if (SynthVoice* v = findActiveVoice())
                v->stopNote(0.0f, true);

            if (baseNote >= 0)
            {
                juce::Synthesiser::handleMidiEvent(
                    juce::MidiMessage::noteOff(baseChannel, baseNote));
            }

            baseNote = -1;
            return;
        }

        juce::Synthesiser::handleMidiEvent(m);
        return;
    }

    //==========================================================================
    if (m.isChannelPressure())
    {
        for (int i = 0; i < getNumVoices(); ++i)
        {
            if (auto* v = dynamic_cast<SynthVoice*>(getVoice(i)))
            {
                if (v->isVoiceActive() && v->isPlayingChannel(m.getChannel()))
                    v->setPressure(static_cast<float>(m.getChannelPressureValue()) / 127.0f);
            }
        }
    }
    
    if (m.isAllNotesOff() || m.isAllSoundOff())
    {
        heldNotes.clear();
        baseNote = -1;
    }

    juce::Synthesiser::handleMidiEvent(m);
}

void StraticSynthesiser::allNotesOff(int midiChannel, bool shouldStopSustainedNotes)
{
    heldNotes.clear();
    baseNote = -1;
    juce::Synthesiser::allNotesOff(midiChannel, shouldStopSustainedNotes);
}

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
StraticSynthAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    const juce::StringArray oscShapes{ "Sine", "Saw", "Square", "Triangle",
                                        "Wavetable", "PWM", "Noise" };

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "osc1Shape", 1 }, "Osc 1 Shape", oscShapes, 1));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "osc2Shape", 1 }, "Osc 2 Shape", oscShapes, 1));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "osc3Shape", 1 }, "Osc 3 Shape", oscShapes, 0));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "osc1Level", 1 }, "Osc 1 Level",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "osc2Level", 1 }, "Osc 2 Level",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "osc3Level", 1 }, "Osc 3 Level",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "detune", 1 }, "Osc 2 Detune",
        juce::NormalisableRange<float>(0.0f, 50.0f, 0.1f), 5.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "osc3Semi", 1 }, "Osc 3 Semi",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 1.0f), -12.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "wtPos", 1 }, "Wavetable Pos",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "wtWarp", 1 }, "WT Warp",
        juce::StringArray{ "Off", "Bend", "Asym", "Quantize", "Smooth" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "wtWarpAmt", 1 }, "WT Warp Amt",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "pwmWidth", 1 }, "PWM Width",
        juce::NormalisableRange<float>(0.05f, 0.95f, 0.01f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fmAmount", 1 }, "FM Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "hardSync", 1 }, "Hard Sync",
        juce::StringArray{ "Off", "On" }, 0));

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "voiceMode", 1 }, "Voice Mode",
        juce::StringArray{ "Poly", "Mono", "Legato" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "glide", 1 }, "Glide",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.001f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{ "uniVoices", 1 }, "Uni Voices", 1, 5, 1));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "uniDetune", 1 }, "Uni Detune",
        juce::NormalisableRange<float>(0.0f, 50.0f, 0.1f), 10.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "uniSpread", 1 }, "Uni Spread",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "filterType", 1 }, "Filter Type",
        juce::StringArray{ "LP", "HP", "BP", "Notch" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "cutoff", 1 }, "Cutoff",
        juce::NormalisableRange<float>(20.0f, 20000.0f, 1.0f, 0.3f), 8000.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "resonance", 1 }, "Resonance",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.2f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "filterDrive", 1 }, "Filter Drive",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "filterKeytrack", 1 }, "Filter Keytrack",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fenvAmount", 1 }, "Fenv Amount",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fenvAttack", 1 }, "Fenv Attack",
        juce::NormalisableRange<float>(0.001f, 5.0f, 0.001f, 0.3f), 0.005f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fenvDecay", 1 }, "Fenv Decay",
        juce::NormalisableRange<float>(0.001f, 5.0f, 0.001f, 0.3f), 0.3f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fenvSustain", 1 }, "Fenv Sustain",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fenvRelease", 1 }, "Fenv Release",
        juce::NormalisableRange<float>(0.001f, 8.0f, 0.001f, 0.3f), 0.3f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "lfoRate", 1 }, "LFO Rate",
        juce::NormalisableRange<float>(0.01f, 20.0f, 0.01f, 0.5f), 2.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "lfoDepth", 1 }, "LFO Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.2f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "lfo2Rate", 1 }, "LFO 2 Rate",
        juce::NormalisableRange<float>(0.01f, 20.0f, 0.01f, 0.5f), 2.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "lfo2Depth", 1 }, "LFO 2 Depth",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "lfo2Shape", 1 }, "LFO 2 Shape",
        juce::StringArray{ "Sine", "Tri", "Saw", "Square", "S&H" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "lfo2Sync", 1 }, "LFO 2 Sync",
        juce::StringArray{ "Free", "1/4", "1/4T", "1/8", "1/8T", "1/16", "1/16T", "1/32" }, 0));

    for (int i = 1; i <= 4; ++i)
    {
        const juce::String idx = juce::String(i);

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{ "modSrc" + idx, 1 }, "Mod Src " + idx,
            juce::StringArray{ "Off", "LFO 1", "LFO 2", "Env Amp", "Env Filt",
                              "ModWheel", "Velocity", "Keytrack", "Random",
                              "Aftertouch", "MPE Timb", "MPE Press" }, 0));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{ "modDest" + idx, 1 }, "Mod Dest " + idx,
            juce::StringArray{ "Off", "WT Pos", "Cutoff", "Resonance",
                              "Osc1 Lvl", "Osc2 Lvl", "Osc3 Lvl",
                              "Osc1 Pch", "Osc2 Pch", "Osc3 Pch",
                              "Volume", "Filt Drv" }, 0));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{ "modAmt" + idx, 1 }, "Mod Amt " + idx,
            juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f));
    }

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "attack", 1 }, "Attack",
        juce::NormalisableRange<float>(0.001f, 5.0f, 0.001f, 0.3f), 0.005f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "decay", 1 }, "Decay",
        juce::NormalisableRange<float>(0.001f, 5.0f, 0.001f, 0.3f), 0.2f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "sustain", 1 }, "Sustain",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.7f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "release", 1 }, "Release",
        juce::NormalisableRange<float>(0.001f, 8.0f, 0.001f, 0.3f), 0.3f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "volume", 1 }, "Volume",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.8f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "sampleLevel", 1 }, "Sample Level",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "sampleMode", 1 }, "Sample Mode",
        juce::StringArray{ "OneShot", "Loop" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{ "sampleRoot", 1 }, "Sample Root", 24, 96, 60));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "sampleStart", 1 }, "Sample Start",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "sampleReverse", 1 }, "Sample Reverse",
        juce::StringArray{ "Off", "On" }, 0));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "fxDrive", 1 }, "Drive",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "chorusRate", 1 }, "Chorus Rate",
        juce::NormalisableRange<float>(0.05f, 5.0f, 0.01f), 0.8f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "chorusDepth", 1 }, "Chorus Depth",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.3f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "chorusMix", 1 }, "Chorus Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "delayTime", 1 }, "Delay Time",
        juce::NormalisableRange<float>(0.0f, 0.75f, 0.001f), 0.25f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "delayFeedback", 1 }, "Delay Feedback",
        juce::NormalisableRange<float>(0.0f, 0.9f, 0.01f), 0.35f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "delayMix", 1 }, "Delay Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "reverbSize", 1 }, "Reverb Size",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "reverbMix", 1 }, "Reverb Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "eqLow", 1 }, "EQ Low",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "eqMid", 1 }, "EQ Mid",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "eqHigh", 1 }, "EQ High",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "compOn", 1 }, "Comp On",
        juce::StringArray{ "Off", "On" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "compThresh", 1 }, "Comp Thresh",
        juce::NormalisableRange<float>(-40.0f, 0.0f, 0.5f), -12.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "compRatio", 1 }, "Comp Ratio",
        juce::NormalisableRange<float>(1.0f, 12.0f, 0.1f), 4.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "compAttack", 1 }, "Comp Attack",
        juce::NormalisableRange<float>(1.0f, 100.0f, 0.5f), 10.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "compRelease", 1 }, "Comp Release",
        juce::NormalisableRange<float>(50.0f, 1000.0f, 1.0f), 200.0f));

    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "arpOn", 1 }, "Arp On",
        juce::StringArray{ "Off", "On" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "arpMode", 1 }, "Arp Mode",
        juce::StringArray{ "Up", "Down", "UpDn", "DnUp", "Random", "Order" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "arpRate", 1 }, "Arp Rate",
        juce::StringArray{ "1/4", "1/8", "1/8T", "1/16", "1/16T", "1/32" }, 3));
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{ "arpOctaves", 1 }, "Arp Octaves", 1, 4, 1));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ "arpGate", 1 }, "Arp Gate",
        juce::NormalisableRange<float>(0.05f, 1.0f, 0.01f), 0.6f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ "mpeOn", 1 }, "MPE On",
        juce::StringArray{ "Off", "On" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        juce::ParameterID{ "mpeBend", 1 }, "MPE Bend", 1, 48, 48));

    return juce::AudioProcessorValueTreeState::ParameterLayout(params.begin(), params.end());
}

//==============================================================================
StraticSynthAudioProcessor::StraticSynthAudioProcessor()
    : juce::AudioProcessor(
        juce::AudioProcessor::BusesProperties()
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
    parameters(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    SynthParamRefs refs;

    refs.cutoff = parameters.getRawParameterValue("cutoff");
    refs.resonance = parameters.getRawParameterValue("resonance");
    refs.attack = parameters.getRawParameterValue("attack");
    refs.decay = parameters.getRawParameterValue("decay");
    refs.sustain = parameters.getRawParameterValue("sustain");
    refs.release = parameters.getRawParameterValue("release");
    refs.detune = parameters.getRawParameterValue("detune");
    refs.volume = parameters.getRawParameterValue("volume");

    refs.lfoRate = parameters.getRawParameterValue("lfoRate");
    refs.lfoDepth = parameters.getRawParameterValue("lfoDepth");
    refs.wtPos = parameters.getRawParameterValue("wtPos");

    refs.osc1Shape = parameters.getRawParameterValue("osc1Shape");
    refs.osc2Shape = parameters.getRawParameterValue("osc2Shape");
    refs.osc3Shape = parameters.getRawParameterValue("osc3Shape");
    refs.osc1Level = parameters.getRawParameterValue("osc1Level");
    refs.osc2Level = parameters.getRawParameterValue("osc2Level");
    refs.osc3Level = parameters.getRawParameterValue("osc3Level");
    refs.osc3Semi = parameters.getRawParameterValue("osc3Semi");

    refs.voiceMode = parameters.getRawParameterValue("voiceMode");
    refs.glide = parameters.getRawParameterValue("glide");

    refs.uniVoices = parameters.getRawParameterValue("uniVoices");
    refs.uniDetune = parameters.getRawParameterValue("uniDetune");
    refs.uniSpread = parameters.getRawParameterValue("uniSpread");

    refs.fxDrive = parameters.getRawParameterValue("fxDrive");
    refs.chorusRate = parameters.getRawParameterValue("chorusRate");
    refs.chorusDepth = parameters.getRawParameterValue("chorusDepth");
    refs.chorusMix = parameters.getRawParameterValue("chorusMix");
    refs.delayTime = parameters.getRawParameterValue("delayTime");
    refs.delayFeedback = parameters.getRawParameterValue("delayFeedback");
    refs.delayMix = parameters.getRawParameterValue("delayMix");
    refs.reverbSize = parameters.getRawParameterValue("reverbSize");
    refs.reverbMix = parameters.getRawParameterValue("reverbMix");

    refs.sampleLevel = parameters.getRawParameterValue("sampleLevel");
    refs.sampleMode = parameters.getRawParameterValue("sampleMode");
    refs.sampleRoot = parameters.getRawParameterValue("sampleRoot");
    refs.sampleStart = parameters.getRawParameterValue("sampleStart");
    refs.sampleReverse = parameters.getRawParameterValue("sampleReverse");

    refs.wavetable = &currentWavetable;

    refs.filterType = parameters.getRawParameterValue("filterType");
    refs.filterDrive = parameters.getRawParameterValue("filterDrive");
    refs.filterKeytrack = parameters.getRawParameterValue("filterKeytrack");
    refs.fenvAmount = parameters.getRawParameterValue("fenvAmount");
    refs.fenvAttack = parameters.getRawParameterValue("fenvAttack");
    refs.fenvDecay = parameters.getRawParameterValue("fenvDecay");
    refs.fenvSustain = parameters.getRawParameterValue("fenvSustain");
    refs.fenvRelease = parameters.getRawParameterValue("fenvRelease");

    refs.wtWarp = parameters.getRawParameterValue("wtWarp");
    refs.wtWarpAmt = parameters.getRawParameterValue("wtWarpAmt");
    
    refs.pwmWidth = parameters.getRawParameterValue("pwmWidth");
    refs.fmAmount = parameters.getRawParameterValue("fmAmount");
    refs.hardSync = parameters.getRawParameterValue("hardSync");

    
    refs.lfo2Rate = parameters.getRawParameterValue("lfo2Rate");
    refs.lfo2Depth = parameters.getRawParameterValue("lfo2Depth");
    refs.lfo2Shape = parameters.getRawParameterValue("lfo2Shape");
    refs.lfo2Sync = parameters.getRawParameterValue("lfo2Sync");

    refs.modSrc1 = parameters.getRawParameterValue("modSrc1");
    refs.modDest1 = parameters.getRawParameterValue("modDest1");
    refs.modAmt1 = parameters.getRawParameterValue("modAmt1");
    refs.modSrc2 = parameters.getRawParameterValue("modSrc2");
    refs.modDest2 = parameters.getRawParameterValue("modDest2");
    refs.modAmt2 = parameters.getRawParameterValue("modAmt2");
    refs.modSrc3 = parameters.getRawParameterValue("modSrc3");
    refs.modDest3 = parameters.getRawParameterValue("modDest3");
    refs.modAmt3 = parameters.getRawParameterValue("modAmt3");
    refs.modSrc4 = parameters.getRawParameterValue("modSrc4");
    refs.modDest4 = parameters.getRawParameterValue("modDest4");
    refs.modAmt4 = parameters.getRawParameterValue("modAmt4");

    refs.lastBpmRef = &lastBpm;

    refs.modWheel = &modWheelValue;
    refs.aftertouch = &aftertouchValue;

    refs.mpeOn = parameters.getRawParameterValue("mpeOn");
    refs.mpeBend = parameters.getRawParameterValue("mpeBend");

    formatManager.registerBasicFormats();

    buildFactoryPresets();
    scanUserPresets();

    synth.params = refs;
    synth.addSound(new SynthSound());

    for (int i = 0; i < 8; ++i)
        synth.addVoice(new SynthVoice(refs));
}

//==============================================================================
void StraticSynthAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    synth.setCurrentPlaybackSampleRate(sampleRate);
    fxChain.prepare(sampleRate);
    fxWork.setSize(2, samplesPerBlock + 32);
}

void StraticSynthAudioProcessor::releaseResources()
{
    fxChain.reset();
}

//==============================================================================
void StraticSynthAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    buffer.clear();

    if (auto* ph = getPlayHead())
    {
        if (auto pos = ph->getPosition())
        {
            if (pos->getBpm().hasValue())
                lastBpm.store(static_cast<float>(*pos->getBpm()),
                    std::memory_order_relaxed);
        }
    }

    for (const auto metadata : midiMessages)
    {
        const auto msg = metadata.getMessage();

        if (msg.isControllerOfType(1))
            modWheelValue.store(static_cast<float>(msg.getControllerValue()) / 127.0f,
                std::memory_order_relaxed);
        else if (msg.isChannelPressure() && msg.getChannel() == 1)
            aftertouchValue.store(static_cast<float>(msg.getChannelPressureValue()) / 127.0f,
                std::memory_order_relaxed);

        if (msg.isController())
        {
            const int cc = msg.getControllerNumber();
            const float v = static_cast<float>(msg.getControllerValue()) / 127.0f;

            const juce::ScopedLock sl(midiMapLock);

            if (pendingLearnParam.isNotEmpty() && cc < 120)
            {
                for (auto it = midiMappings.begin(); it != midiMappings.end();)
                {
                    if (it->paramID == pendingLearnParam || it->cc == cc)
                        it = midiMappings.erase(it);
                    else
                        ++it;
                }

                midiMappings.push_back({ pendingLearnParam, cc });
                pendingLearnParam.clear();
            }
            else if (cc < 120)
            {
                for (const auto& m : midiMappings)
                {
                    if (m.cc == cc)
                    {
                        if (auto* rp = dynamic_cast<juce::RangedAudioParameter*>(
                            parameters.getParameter(m.paramID)))
                            rp->setValue(v);
                    }
                }
            }
        }
    }

    //==========================================================================
    // Arpeggiator
    //==========================================================================
    const bool arpOnNow =
        static_cast<int>(parameters.getRawParameterValue("arpOn")->load()) == 1;
    const int arpMode =
        static_cast<int>(parameters.getRawParameterValue("arpMode")->load());
    const int arpRateIdx =
        static_cast<int>(parameters.getRawParameterValue("arpRate")->load());
    const int arpOct = juce::jlimit(
        1, 4, static_cast<int>(parameters.getRawParameterValue("arpOctaves")->load()));
    const float arpGate = parameters.getRawParameterValue("arpGate")->load();

    juce::MidiBuffer playMidi;

    if (!arpOnNow)
    {
        if (lastArpNote >= 0)
        {
            playMidi.addEvent(juce::MidiMessage::noteOff(1, lastArpNote), 0);
            lastArpNote = -1;
        }

        heldArp.clear();
        nextStepAbs = 0.0;

        for (const auto m : midiMessages)
            playMidi.addEvent(m.getMessage(), m.samplePosition);
    }
    else
    {
        for (const auto m : midiMessages)
        {
            const auto msg = m.getMessage();

            if (msg.isNoteOn())
            {
                if (std::find(heldArp.begin(), heldArp.end(), msg.getNoteNumber()) == heldArp.end())
                    heldArp.push_back(msg.getNoteNumber());

                lastArpVel = msg.getVelocity();
            }
            else if (msg.isNoteOff())
            {
                heldArp.erase(std::remove(heldArp.begin(), heldArp.end(),
                    msg.getNoteNumber()), heldArp.end());
            }
            else
            {
                playMidi.addEvent(msg, m.samplePosition);
            }
        }

        const double sr = getSampleRate();
        const double bpm = juce::jmax(30.0, static_cast<double>(lastBpm.load()));

        static const double factors[6] = { 1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 6.0, 0.125 };
        const double stepDur = (60.0 / bpm) * factors[juce::jlimit(0, 5, arpRateIdx)] * sr;
        const double gateSamples = stepDur * juce::jlimit(0.05, 1.0, static_cast<double>(arpGate));

        const double blockStart = absoluteSamplePos;
        const double blockEnd = blockStart + buffer.getNumSamples();

        if (nextStepAbs < blockStart)
            nextStepAbs = blockStart;

        while (nextStepAbs < blockEnd && !heldArp.empty())
        {
            const auto seq = buildArpSeq(arpMode, arpOct);

            if (seq.empty())
                break;

            int note;

            if (arpMode == 4)
                note = seq[static_cast<size_t>(
                    juce::Random::getSystemRandom().nextInt(static_cast<int>(seq.size())))];
            else
                note = seq[static_cast<size_t>(arpStep % seq.size())];

            ++arpStep;

            const int onPos = juce::jlimit(0, buffer.getNumSamples() - 1,
                static_cast<int>(nextStepAbs - blockStart));

            if (lastArpNote >= 0)
            {
                const int offPos = juce::jlimit(0, onPos,
                    static_cast<int>(pendingOffAbs - blockStart));
                playMidi.addEvent(juce::MidiMessage::noteOff(1, lastArpNote), offPos);
            }

            playMidi.addEvent(juce::MidiMessage::noteOn(1, note,
                static_cast<juce::uint8>(lastArpVel)), onPos);

            lastArpNote = note;
            pendingOffAbs = nextStepAbs + gateSamples;

            nextStepAbs += stepDur;
        }

        if (heldArp.empty() && lastArpNote >= 0 && pendingOffAbs <= blockEnd)
        {
            const int offPos = juce::jlimit(0, buffer.getNumSamples() - 1,
                static_cast<int>(pendingOffAbs - blockStart));
            playMidi.addEvent(juce::MidiMessage::noteOff(1, lastArpNote), offPos);
            lastArpNote = -1;
        }
    }

    absoluteSamplePos += buffer.getNumSamples();

    synth.renderNextBlock(buffer, playMidi, 0, buffer.getNumSamples());

    const float gain = parameters.getRawParameterValue("volume")->load();
    buffer.applyGain(gain);

    FxParams fp;
    fp.drive = parameters.getRawParameterValue("fxDrive")->load();
    fp.chorusRate = parameters.getRawParameterValue("chorusRate")->load();
    fp.chorusDepth = parameters.getRawParameterValue("chorusDepth")->load();
    fp.chorusMix = parameters.getRawParameterValue("chorusMix")->load();
    fp.delayTime = parameters.getRawParameterValue("delayTime")->load();
    fp.delayFeedback = parameters.getRawParameterValue("delayFeedback")->load();
    fp.delayMix = parameters.getRawParameterValue("delayMix")->load();
    fp.reverbSize = parameters.getRawParameterValue("reverbSize")->load();
    fp.reverbMix = parameters.getRawParameterValue("reverbMix")->load();

    fp.eqLow = parameters.getRawParameterValue("eqLow")->load();
    fp.eqMid = parameters.getRawParameterValue("eqMid")->load();
    fp.eqHigh = parameters.getRawParameterValue("eqHigh")->load();
    fp.compOn = static_cast<int>(parameters.getRawParameterValue("compOn")->load()) == 1;
    fp.compThresh = parameters.getRawParameterValue("compThresh")->load();
    fp.compRatio = parameters.getRawParameterValue("compRatio")->load();
    fp.compAttack = parameters.getRawParameterValue("compAttack")->load();
    fp.compRelease = parameters.getRawParameterValue("compRelease")->load();

    if (buffer.getNumChannels() >= 2)
    {
        fxChain.process(buffer, fp);
    }
    else if (buffer.getNumChannels() == 1 &&
        fxWork.getNumSamples() >= buffer.getNumSamples())
    {
        const int n = buffer.getNumSamples();

        fxWork.copyFrom(0, 0, buffer, 0, 0, n);
        fxWork.copyFrom(1, 0, buffer, 0, 0, n);

        fxChain.process(fxWork, fp);

        buffer.copyFrom(0, 0, fxWork, 0, 0, n);
    }

    {
        const int n = buffer.getNumSamples();
        const float* l = buffer.getReadPointer(0);
        const float* r = buffer.getNumChannels() > 1 ? buffer.getReadPointer(1) : l;

        float pl = 0.0f, pr = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            pl = juce::jmax(pl, std::abs(l[i]));
            pr = juce::jmax(pr, std::abs(r[i]));
        }

        monitor.pushBlock(l, r, n);
        monitor.pushPeaks(pl, pr);
    }
}

//==============================================================================
bool StraticSynthAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() ||
        out == juce::AudioChannelSet::stereo();
}

bool StraticSynthAudioProcessor::hasEditor() const { return true; }

const juce::String StraticSynthAudioProcessor::getName() const { return "Stratic Synth"; }
bool StraticSynthAudioProcessor::acceptsMidi() const { return true; }
bool StraticSynthAudioProcessor::producesMidi() const { return false; }
bool StraticSynthAudioProcessor::isMidiEffect() const { return false; }
double StraticSynthAudioProcessor::getTailLengthSeconds() const { return 3.0; }

int StraticSynthAudioProcessor::getNumPrograms()
{
    return static_cast<int>(factoryPresets.size());
}

int StraticSynthAudioProcessor::getCurrentProgram()
{
    return currentPresetIndex < 0 ? 0 : currentPresetIndex;
}

void StraticSynthAudioProcessor::setCurrentProgram(int index)
{
    loadPresetByIndex(index);
}

const juce::String StraticSynthAudioProcessor::getProgramName(int index)
{
    return getPresetName(index);
}

void StraticSynthAudioProcessor::changeProgramName(int, const juce::String&) {}

//==============================================================================
void StraticSynthAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto tree = parameters.copyState();

    juce::ValueTree learnTree("midiLearn");

    {
        const juce::ScopedLock sl(midiMapLock);

        for (const auto& m : midiMappings)
        {
            juce::ValueTree e("map");
            e.setProperty("id", m.paramID, nullptr);
            e.setProperty("cc", m.cc, nullptr);
            learnTree.addChild(e, -1, nullptr);
        }
    }

    tree.addChild(learnTree, -1, nullptr);

    if (auto xml = tree.createXml())
        copyXmlToBinary(*xml, destData);
}

void StraticSynthAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
    {
        auto tree = juce::ValueTree::fromXml(*xml);

        if (tree.isValid() && tree.getType() == parameters.state.getType())
        {
            const auto learnTree = tree.getChildWithName("midiLearn");

            {
                const juce::ScopedLock sl(midiMapLock);
                midiMappings.clear();

                if (learnTree.isValid())
                {
                    for (int i = 0; i < learnTree.getNumChildren(); ++i)
                    {
                        const auto e = learnTree.getChild(i);
                        midiMappings.push_back({ e.getProperty("id").toString(),
                                                 static_cast<int>(e.getProperty("cc", -1)) });
                    }
                }
            }

            parameters.replaceState(tree);
        }
    }
}

//==============================================================================
int StraticSynthAudioProcessor::getNumPresets() const
{
    return static_cast<int>(factoryPresets.size() + userPresets.size());
}

juce::String StraticSynthAudioProcessor::getPresetName(int index) const
{
    const int nf = static_cast<int>(factoryPresets.size());

    if (index < 0)
        return {};

    if (index < nf)
        return factoryPresets[static_cast<size_t>(index)].name;

    const int ui = index - nf;

    if (ui < static_cast<int>(userPresets.size()))
        return userPresets[static_cast<size_t>(ui)].name;

    return {};
}

void StraticSynthAudioProcessor::loadPresetByIndex(int index)
{
    const int nf = static_cast<int>(factoryPresets.size());
    const Preset* p = nullptr;

    if (index >= 0 && index < nf)
        p = &factoryPresets[static_cast<size_t>(index)];
    else if (index >= nf && index - nf < static_cast<int>(userPresets.size()))
        p = &userPresets[static_cast<size_t>(index - nf)];

    if (p == nullptr)
        return;

    currentPresetIndex = index;
    parameters.replaceState(p->state.createCopy());
}

int StraticSynthAudioProcessor::getCurrentPresetIndex() const { return currentPresetIndex; }

juce::String StraticSynthAudioProcessor::getCurrentPresetName() const
{
    return parameters.state.getProperty("presetName", "Init").toString();
}

void StraticSynthAudioProcessor::nextPreset()
{
    const int n = static_cast<int>(factoryPresets.size());
    if (n <= 0) return;
    loadPresetByIndex((currentPresetIndex + 1) % n);
}

void StraticSynthAudioProcessor::prevPreset()
{
    const int n = static_cast<int>(factoryPresets.size());
    if (n <= 0) return;
    loadPresetByIndex((currentPresetIndex - 1 + n) % n);
}

bool StraticSynthAudioProcessor::savePresetToFile(const juce::File& file,
    const juce::String& name)
{
    auto tree = parameters.copyState();
    tree.setProperty("presetName", name, nullptr);

    if (auto xml = tree.createXml())
        return xml->writeTo(file);

    return false;
}

//==============================================================================
juce::File StraticSynthAudioProcessor::getUserPresetDirectory() const
{
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("StraticSynthPresets");
}

//==============================================================================
void StraticSynthAudioProcessor::scanUserPresets()
{
    userPresets.clear();

    auto dir = getUserPresetDirectory();

    if (!dir.isDirectory())
        return;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.straticp");

    std::sort(files.begin(), files.end(),
        [](const juce::File& a, const juce::File& b)
        {
            return a.getFileNameWithoutExtension()
                .compareNatural(b.getFileNameWithoutExtension()) < 0;
        });

    for (const auto& f : files)
    {
        auto xml = juce::XmlDocument::parse(f);

        if (xml == nullptr)
            continue;

        auto tree = juce::ValueTree::fromXml(*xml);

        if (!tree.isValid() || tree.getType() != parameters.state.getType())
            continue;

        userPresets.push_back({ f.getFileNameWithoutExtension(), tree });
    }
}

//==============================================================================
void StraticSynthAudioProcessor::saveCurrentAsUserPreset(const juce::String& name)
{
    auto dir = getUserPresetDirectory();
    dir.createDirectory();

    auto tree = parameters.copyState();
    tree.setProperty("presetName", name, nullptr);

    const auto file = dir.getChildFile(name + ".straticp");

    if (auto xml = tree.createXml())
        xml->writeTo(file);

    scanUserPresets();

    const int nf = static_cast<int>(factoryPresets.size());

    for (int i = 0; i < static_cast<int>(userPresets.size()); ++i)
    {
        if (userPresets[static_cast<size_t>(i)].name == name)
        {
            currentPresetIndex = nf + i;
            break;
        }
    }
}

bool StraticSynthAudioProcessor::loadPresetFromFile(const juce::File& file)
{
    auto xml = juce::XmlDocument::parse(file);
    if (xml == nullptr) return false;

    auto tree = juce::ValueTree::fromXml(*xml);
    if (!tree.isValid() || tree.getType() != parameters.state.getType())
        return false;

    currentPresetIndex = -1;
    parameters.replaceState(tree);
    return true;
}

//==============================================================================
bool StraticSynthAudioProcessor::loadSampleFromFile(const juce::File& file)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr) return false;

    const std::int64_t cap = static_cast<std::int64_t>(reader->sampleRate * 10.0);
    const std::int64_t total64 = reader->lengthInSamples < cap ? reader->lengthInSamples : cap;
    const int total = static_cast<int>(total64);

    if (total < 128) return false;

    juce::AudioBuffer<float> src(static_cast<int>(reader->numChannels), total);
    src.clear();
    reader->read(&src, 0, total, 0, true, true);

    std::vector<float> mono(static_cast<size_t>(total), 0.0f);

    if (src.getNumChannels() >= 2)
    {
        const float* l = src.getReadPointer(0);
        const float* r = src.getReadPointer(1);
        for (int i = 0; i < total; ++i)
            mono[static_cast<size_t>(i)] = (l[i] + r[i]) * 0.5f;
    }
    else
    {
        const float* l = src.getReadPointer(0);
        for (int i = 0; i < total; ++i)
            mono[static_cast<size_t>(i)] = l[i];
    }

    auto data = std::make_unique<SampleData>();
    data->buffer.setSize(1, total);
    data->buffer.copyFrom(0, 0, mono.data(), total);
    data->sampleRate = reader->sampleRate;

    sampleHistory.push_back(std::move(data));
    currentSample.store(sampleHistory.back().get());

    sampleName = file.getFileNameWithoutExtension();
    return true;
}

juce::String StraticSynthAudioProcessor::getSampleName() const { return sampleName; }

//==============================================================================
bool StraticSynthAudioProcessor::loadWavetableFromFile(const juce::File& file)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr) return false;

    const std::int64_t cap = static_cast<std::int64_t>(reader->sampleRate * 10.0);
    const int total = static_cast<int>(
        reader->lengthInSamples < cap ? reader->lengthInSamples : cap);

    if (total < 128) return false;

    juce::AudioBuffer<float> src(static_cast<int>(reader->numChannels), total);
    src.clear();
    reader->read(&src, 0, total, 0, true, true);

    std::vector<float> mono(static_cast<size_t>(total), 0.0f);

    if (src.getNumChannels() >= 2)
    {
        const float* l = src.getReadPointer(0);
        const float* r = src.getReadPointer(1);
        for (int i = 0; i < total; ++i)
            mono[static_cast<size_t>(i)] = (l[i] + r[i]) * 0.5f;
    }
    else
    {
        const float* l = src.getReadPointer(0);
        for (int i = 0; i < total; ++i)
            mono[static_cast<size_t>(i)] = l[i];
    }

    int frameSize = 0;

    for (int fs : { 2048, 1024, 512, 256, 128 })
    {
        const int count = total / fs;
        if (total % fs == 0 && count >= 2 && count <= 128)
        {
            frameSize = fs;
            break;
        }
    }

    int numFrames = 0;

    if (frameSize > 0)
        numFrames = total / frameSize;
    else
    {
        numFrames = 32;
        frameSize = total / numFrames;
    }

    if (frameSize < 32 || numFrames < 2) return false;

    auto data = std::make_unique<WavetableData>();
    data->frames.clear();
    data->tableSize = 1024;
    data->numFrames = numFrames;
    data->frames.resize(static_cast<size_t>(numFrames),
        std::vector<float>(1024, 0.0f));

    for (int f = 0; f < numFrames; ++f)
    {
        const double frameStart = static_cast<double>(f * frameSize);

        for (int i = 0; i < 1024; ++i)
        {
            const double pos = frameStart +
                (static_cast<double>(i) / 1024.0) * static_cast<double>(frameSize);

            int i0 = juce::jlimit(0, total - 1, static_cast<int>(pos));
            const int i1 = juce::jmin(total - 1, i0 + 1);
            const float frac = static_cast<float>(pos - static_cast<double>(i0));

            data->frames[static_cast<size_t>(f)][static_cast<size_t>(i)] =
                mono[static_cast<size_t>(i0)] +
                (mono[static_cast<size_t>(i1)] - mono[static_cast<size_t>(i0)]) * frac;
        }
    }

    for (int f = 0; f < numFrames; ++f)
        data->normalizeFrame(f);

    wtHistory.push_back(std::move(data));
    currentWavetable.store(wtHistory.back().get());

    return true;
}

void StraticSynthAudioProcessor::resetWavetable()
{
    currentWavetable.store(nullptr);
}

//==============================================================================
void StraticSynthAudioProcessor::applyWavetable(const WavetableData& data)
{
    auto copy = std::make_unique<WavetableData>();

    copy->frames = data.frames;
    copy->tableSize = data.tableSize;
    copy->numFrames = data.numFrames;

    wtHistory.push_back(std::move(copy));
    currentWavetable.store(wtHistory.back().get());
}

//==============================================================================
void StraticSynthAudioProcessor::applyPresetValues(
    std::initializer_list<std::pair<const char*, double>> values)
{
    for (const auto& v : values)
    {
        if (auto* rp = dynamic_cast<juce::RangedAudioParameter*>(
            parameters.getParameter(v.first)))
        {
            rp->setValueNotifyingHost(
                rp->convertTo0to1(static_cast<float>(v.second)));
        }
    }
}

void StraticSynthAudioProcessor::buildFactoryPresets()
{
    auto snapshot = [this](const juce::String& name)
        {
            auto tree = parameters.copyState();
            tree.setProperty("presetName", name, nullptr);
            factoryPresets.push_back({ name, tree });
        };

#define P_FULL(name, o1,o2,o3, l1,l2,l3, det, semi, wt, cut, res, lr, ld, \
                   a,d,s,r, vol, vm, gl, uv, ud, us, drv, cr,cd,cm, dt,df,dm, rs,rm) \
        applyPresetValues({ \
            { "osc1Shape", o1 }, { "osc2Shape", o2 }, { "osc3Shape", o3 }, \
            { "osc1Level", l1 }, { "osc2Level", l2 }, { "osc3Level", l3 }, \
            { "detune", det }, { "osc3Semi", semi }, { "wtPos", wt }, \
            { "cutoff", cut }, { "resonance", res }, \
            { "lfoRate", lr }, { "lfoDepth", ld }, \
            { "attack", a }, { "decay", d }, { "sustain", s }, { "release", r }, \
            { "volume", vol }, { "voiceMode", vm }, { "glide", gl }, \
            { "uniVoices", uv }, { "uniDetune", ud }, { "uniSpread", us }, \
            { "fxDrive", drv }, \
            { "chorusRate", cr }, { "chorusDepth", cd }, { "chorusMix", cm }, \
            { "delayTime", dt }, { "delayFeedback", df }, { "delayMix", dm }, \
            { "reverbSize", rs }, { "reverbMix", rm }, \
            { "filterType", 0 }, { "filterDrive", 0 }, { "filterKeytrack", 0 }, \
            { "fenvAmount", 0 }, \
            { "lfo2Rate", 2 }, { "lfo2Depth", 0 }, { "lfo2Shape", 0 }, { "lfo2Sync", 0 }, \
            { "modSrc1", 0 }, { "modDest1", 0 }, { "modAmt1", 0 }, \
            { "modSrc2", 0 }, { "modDest2", 0 }, { "modAmt2", 0 }, \
            { "modSrc3", 0 }, { "modDest3", 0 }, { "modAmt3", 0 }, \
            { "modSrc4", 0 }, { "modDest4", 0 }, { "modAmt4", 0 }, \
            { "wtWarp", 0 }, { "wtWarpAmt", 0.5 }, \
            { "pwmWidth", 0.5 }, { "fmAmount", 0 }, { "hardSync", 0 }, \
            { "eqLow", 0 }, { "eqMid", 0 }, { "eqHigh", 0 }, \
            { "compOn", 0 }, { "compThresh", -12 }, { "compRatio", 4 }, \
            { "compAttack", 10 }, { "compRelease", 200 }, \
            { "arpOn", 0 }, { "arpMode", 0 }, { "arpRate", 3 }, \
            { "arpOctaves", 1 }, { "arpGate", 0.6 }, \
            { "mpeOn", 0 }, { "mpeBend", 48 } }); \
        snapshot(name);

    P_FULL("Init", 1, 1, 0, .5, .5, 0, 5, -12, .5, 8000, .2, 2, .2, .005, .2, .7, .3, .8, 0, 0, 1, 10, .5, 0, .8, .3, 0, .25, .35, 0, .5, 0)
        P_FULL("Sub Bass", 1, 2, 0, .5, .25, .6, 8, -12, .5, 600, .3, .5, 0, .001, .25, .5, .12, .85, 1, .06, 1, 10, .5, .15, .8, .3, 0, .25, .35, 0, .5, 0)
        P_FULL("Acid Lead", 1, 2, 0, .6, .35, 0, 12, -12, .6, 1400, .7, 6, .35, .001, .18, .3, .15, .8, 1, .08, 1, 10, .5, .2, .8, .3, 0, .25, .35, 0, .5, 0)
        P_FULL("Warm Pad", 3, 1, 0, .5, .35, .3, 18, 0, .3, 2500, .15, .5, .3, .8, 1.5, .8, 2.5, .8, 0, 0, 3, 12, .6, 0, .6, .4, .35, .25, .35, 0, .7, .4)
        P_FULL("Wavetable Keys", 4, 4, 0, .6, .3, 0, 6, -12, .35, 4000, .2, 1, .1, .002, .6, .4, .8, .8, 0, 0, 1, 10, .5, 0, .8, .3, .25, .25, .35, 0, .5, .25)
        P_FULL("Pluck", 4, 1, 0, .7, .2, 0, 10, -12, .8, 3000, .5, 2, 0, .001, .35, .05, .3, .8, 0, 0, 1, 10, .5, 0, .8, .3, 0, .25, .35, .3, .5, .3)
        P_FULL("Reese Bass", 1, 1, 0, .5, .5, .4, 35, -12, .5, 900, .25, .8, .15, .005, .3, .6, .2, .85, 1, .04, 2, 20, .4, .3, .8, .3, 0, .25, .35, 0, .5, 0)
        P_FULL("Supersaw", 1, 1, 1, .5, .4, .3, 15, 0, .5, 6000, .15, 3, .1, .002, .3, .7, .4, .75, 0, 0, 5, 18, .8, 0, .7, .3, .3, .28, .3, .25, .6, .3)
        P_FULL("Trance Pluck", 1, 2, 0, .6, .3, 0, 8, -12, .5, 2500, .6, 2, 0, .001, .25, .1, .35, .8, 0, 0, 2, 10, .5, 0, .8, .3, 0, .25, .45, .4, .6, .35)
        P_FULL("Ambient Pad", 4, 3, 0, .5, .3, .3, 20, 0, .2, 1800, .1, .25, .45, 1.5, 2, .9, 4, .75, 0, 0, 4, 15, .9, 0, .4, .5, .45, .4, .3, .2, .85, .55)
        P_FULL("Glass Bell", 0, 0, 0, .5, .25, .35, 3, 12, .5, 8000, .1, 5, .05, .001, 1.2, .15, 1.5, .8, 0, 0, 1, 10, .5, 0, .8, .3, 0, .3, .25, .2, .7, .4)
        P_FULL("Combo Organ", 0, 0, 0, .5, .35, .25, 4, 12, .5, 5000, .05, 6, .08, .01, .1, .9, .15, .8, 0, 0, 1, 10, .5, .1, .9, .25, .3, .25, .35, 0, .5, .2)
        P_FULL("Analog Strings", 1, 1, 0, .5, .5, 0, 12, -12, .5, 3000, .1, .6, .2, .4, .5, .8, 1.2, .8, 0, 0, 3, 10, .7, 0, .5, .4, .4, .25, .35, 0, .7, .35)
        P_FULL("Wobble Bass", 1, 2, 0, .5, .3, .5, 6, -12, .5, 800, .55, 4, .8, .001, .2, .6, .15, .85, 1, .05, 1, 10, .5, .25, .8, .3, 0, .25, .35, 0, .5, 0)
        P_FULL("Seq Arp", 2, 1, 0, .5, .4, 0, 5, -12, .5, 3500, .35, 2, 0, .001, .15, .2, .12, .8, 1, 0, 1, 10, .5, 0, .8, .3, 0, .18, .4, .35, .5, .25)
        P_FULL("LoFi Keys", 4, 3, 0, .6, .3, 0, 9, -12, .5, 1500, .15, 1.2, .15, .005, .4, .5, .5, .8, 0, 0, 1, 10, .5, .35, .6, .3, .3, .3, .3, .2, .6, .3)

        P_FULL("Rock Power Chord", 1, 1, 1, .5, .4, .35, 10, 7, .5, 4000, .2, 2, 0, .003, .2, .9, .25, .8, 0, 0, 2, 12, .5, .5, .8, .3, 0, .25, .35, 0, .5, .2)
        P_FULL("Rock Lead Guitar", 1, 2, 0, .6, .35, 0, 6, -12, .5, 5000, .3, 5, .06, .004, .3, .8, .3, .8, 1, .02, 2, 8, .4, .6, .9, .25, .2, .22, .3, .3, .6, .35)
        P_FULL("Hard Rock Riff", 2, 1, 0, .55, .4, 0, 14, -12, .5, 2500, .4, 3, 0, .002, .25, .7, .18, .85, 1, .01, 2, 15, .4, .7, .8, .3, 0, .25, .35, 0, .5, .15)
        P_FULL("Rock Ballad Keys", 3, 0, 0, .6, .4, 0, 4, -12, .5, 4500, .1, 1, .1, .01, .5, .8, .8, .8, 0, 0, 1, 10, .5, .1, .7, .35, .35, .3, .3, .25, .6, .4)
        P_FULL("Stadium Pad", 1, 1, 0, .5, .5, 0, 18, -12, .5, 3000, .12, .4, .3, .3, 1, .9, 2.5, .78, 0, 0, 4, 14, .8, .05, .5, .45, .4, .35, .35, .3, .8, .5)
        P_FULL("Garage Organ", 2, 0, 0, .55, .45, 0, 3, 12, .5, 5000, .08, 6.5, .1, .008, .15, 1, .12, .8, 0, 0, 1, 10, .5, .4, .9, .3, .4, .25, .35, 0, .5, .25)
        P_FULL("Rock Bass Pick", 1, 2, 0, .5, .35, .4, 5, -12, .5, 1200, .3, 1, 0, .002, .25, .6, .12, .85, 1, .02, 1, 10, .5, .5, .8, .3, 0, .25, .35, 0, .4, .1)
        P_FULL("Anthem Saw Lead", 1, 1, 0, .5, .5, 0, 12, -12, .5, 5500, .18, 2.5, .12, .005, .3, .9, .5, .78, 0, 0, 4, 16, .75, .25, .6, .35, .3, .27, .35, .35, .65, .4)
        P_FULL("Strum Pluck", 3, 0, 0, .6, .4, 0, 7, -12, .5, 3500, .12, 1.5, 0, .001, .4, .05, .4, .8, 0, 0, 1, 10, .5, .1, .8, .3, .25, .24, .3, .2, .55, .3)
        P_FULL("Rock Synth Brass", 1, 2, 0, .5, .4, 0, 9, -12, .5, 4500, .35, 4, .08, .012, .2, .8, .22, .8, 0, 0, 2, 10, .5, .2, .7, .3, .15, .2, .25, .15, .55, .25)

        P_FULL("Horror Fog Drone", 1, 3, 0, .35, .25, 0, 30, -12, .3, 400, .2, .1, .5, 3, 2, 1, 6, .75, 0, 0, 3, 30, 1, .15, .3, .6, .4, .5, .4, .2, .9, .7)
        P_FULL("Horror Music Box", 0, 0, 0, .5, .2, 0, 8, 12, .5, 6000, .05, 2, 0, .001, 1.5, 0, 1.2, .8, 0, 0, 1, 10, .5, .05, .6, .2, .15, .35, .4, .3, .8, .6)
        P_FULL("Horror Rusty Bell", 0, 2, 0, .5, .12, 0, 11, 12, .5, 5000, .15, .3, .2, .001, 2, .05, 2.5, .78, 0, 0, 1, 10, .5, .15, .4, .4, .2, .42, .45, .3, .85, .6)
        P_FULL("Horror Industrial Pulse", 2, 1, 0, .5, .3, 0, 4, -12, .5, 600, .6, 8, .6, .001, .18, .3, .1, .8, 1, 0, 1, 10, .5, .55, .8, .3, 0, .12, .3, .15, .6, .3)
        P_FULL("Horror Silent Hill Pad", 1, 1, 0, .45, .45, 0, 40, 0, .4, 800, .18, .15, .6, 2, 1.5, 1, 5, .75, 0, 0, 5, 40, .95, .1, .3, .6, .5, .45, .4, .25, .9, .65)
        P_FULL("Horror Ghost Choir", 4, 3, 0, .5, .3, 0, 25, 0, .6, 2000, .12, .5, .35, 1.2, 1, .9, 4, .75, 0, 0, 4, 20, .9, 0, .4, .5, .45, .4, .35, .25, .85, .6)
        P_FULL("Horror Radio Keys", 4, 3, 0, .6, .25, 0, 9, 0, .5, 1000, .2, 7, .15, .004, .5, .35, .6, .8, 0, 0, 1, 10, .5, .5, .6, .3, .2, .2, .3, .25, .5, .4)
        P_FULL("Horror Dread Sub", 0, 0, 0, .4, .15, .6, 5, -24, .5, 200, .25, .2, .3, .5, 1, 1, 4, .85, 1, .15, 1, 10, .5, .2, .8, .3, 0, .25, .35, 0, .6, .3)
        P_FULL("Horror Metallic Seq", 2, 1, 0, .5, .35, 0, 6, 0, .5, 1500, .7, 6, .5, .001, .2, .1, .15, .8, 1, 0, 1, 10, .5, .4, .8, .3, 0, .15, .5, .4, .7, .4)
        P_FULL("Horror Melancholy Piano", 3, 0, 4, .5, .3, .15, 2, 0, .3, 5000, .08, 1.5, .06, .002, 1, .25, 1, .8, 0, 0, 1, 10, .5, .1, .5, .25, .15, .3, .25, .2, .7, .45)

#undef P_FULL

        loadPresetByIndex(0);
}

//==============================================================================
std::vector<int> StraticSynthAudioProcessor::buildArpSeq(int mode, int octaves) const
{
    std::vector<int> seq;

    if (heldArp.empty())
        return seq;

    std::vector<int> sorted = heldArp;
    std::sort(sorted.begin(), sorted.end());

    auto appendOct = [&](const std::vector<int>& src)
        {
            for (int o = 0; o < octaves; ++o)
                for (int n : src)
                    seq.push_back(n + 12 * o);
        };

    switch (mode)
    {
    case 1: // Down
    {
        std::vector<int> rev(sorted.rbegin(), sorted.rend());
        appendOct(rev);
        break;
    }
    case 2: // UpDown
    {
        appendOct(sorted);
        if (sorted.size() > 1)
        {
            std::vector<int> mid(sorted.rbegin() + 1, sorted.rend() - 1);
            for (int o = 0; o < octaves; ++o)
                for (int n : mid) seq.push_back(n + 12 * o);
        }
        break;
    }
    case 3: // DownUp
    {
        std::vector<int> rev(sorted.rbegin(), sorted.rend());
        appendOct(rev);
        if (sorted.size() > 1)
        {
            std::vector<int> mid(sorted.begin() + 1, sorted.end() - 1);
            for (int o = 0; o < octaves; ++o)
                for (int n : mid) seq.push_back(n + 12 * o);
        }
        break;
    }
    case 5: // Order (как нажимали)
        appendOct(heldArp);
        break;
    default: // Up / Random
        appendOct(sorted);
        break;
    }

    return seq;
}

//==============================================================================
void StraticSynthAudioProcessor::startMidiLearn(const juce::String& paramID)
{
    const juce::ScopedLock sl(midiMapLock);
    pendingLearnParam = paramID;
}

void StraticSynthAudioProcessor::cancelMidiLearn()
{
    const juce::ScopedLock sl(midiMapLock);
    pendingLearnParam.clear();
}

bool StraticSynthAudioProcessor::isMidiLearnPending() const
{
    const juce::ScopedLock sl(midiMapLock);
    return pendingLearnParam.isNotEmpty();
}

int StraticSynthAudioProcessor::getMappingCC(const juce::String& paramID) const
{
    const juce::ScopedLock sl(midiMapLock);

    for (const auto& m : midiMappings)
        if (m.paramID == paramID)
            return m.cc;

    return -1;
}

void StraticSynthAudioProcessor::clearMapping(const juce::String& paramID)
{
    const juce::ScopedLock sl(midiMapLock);

    for (auto it = midiMappings.begin(); it != midiMappings.end(); ++it)
    {
        if (it->paramID == paramID)
        {
            midiMappings.erase(it);
            break;
        }
    }
}

std::vector<std::pair<juce::String, int>> StraticSynthAudioProcessor::getMappings() const
{
    const juce::ScopedLock sl(midiMapLock);

    std::vector<std::pair<juce::String, int>> out;

    for (const auto& m : midiMappings)
        out.push_back({ m.paramID, m.cc });

    return out;
}

//==============================================================================
bool StraticSynthAudioProcessor::exportBankToFile(const juce::File& file)
{
    juce::ValueTree bank("StraticBank");

    auto add = [&](const Preset& p)
        {
            juce::ValueTree e("preset");
            e.setProperty("name", p.name, nullptr);
            e.addChild(p.state.createCopy(), -1, nullptr);
            bank.addChild(e, -1, nullptr);
        };

    for (const auto& p : factoryPresets)
        add(p);

    for (const auto& p : userPresets)
        add(p);

    if (auto xml = bank.createXml())
        return xml->writeTo(file);

    return false;
}

//==============================================================================
bool StraticSynthAudioProcessor::importBankFromFile(const juce::File& file)
{
    auto xml = juce::XmlDocument::parse(file);

    if (xml == nullptr)
        return false;

    auto bank = juce::ValueTree::fromXml(*xml);

    if (!bank.isValid() || bank.getType() != juce::Identifier("StraticBank"))
        return false;

    std::vector<Preset> imported;

    for (int i = 0; i < bank.getNumChildren(); ++i)
    {
        const auto e = bank.getChild(i);

        if (e.getType() != juce::Identifier("preset"))
            continue;

        const auto name = e.getProperty("name").toString();

        if (name.isEmpty() || e.getNumChildren() < 1)
            continue;

        auto state = e.getChild(0);

        if (state.getType() != parameters.state.getType())
            continue;

        bool isFactory = false;

        for (const auto& f : factoryPresets)
            if (f.name == name) { isFactory = true; break; }

        if (!isFactory)
            imported.push_back({ name, state.createCopy() });
    }

    //==========================================================================
    // заменяем пользовательский банк на диске
    auto dir = getUserPresetDirectory();
    dir.createDirectory();

    juce::Array<juce::File> old;
    dir.findChildFiles(old, juce::File::findFiles, false, "*.straticp");

    for (auto& f : old)
        f.deleteFile();

    for (const auto& p : imported)
    {
        const auto pf = dir.getChildFile(p.name + ".straticp");

        if (auto px = p.state.createXml())
            px->writeTo(pf);
    }

    scanUserPresets();
    return true;
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new StraticSynthAudioProcessor();
}