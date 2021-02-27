import { FuncCanvas } from "./funcCanvas";

export class Synth {
  audioCtx: AudioContext;
  oscilatorNode: OscillatorNode;
  lowPassNode: BiquadFilterNode;
  gainNode: GainNode;

  adsFunc: FuncCanvas;
  rFunc: FuncCanvas;

  lowPassFunc: FuncCanvas;

  constructor() {
    console.log("Constructing synth...");
    const body = document.getElementsByTagName('body')[0];
    const div = document.createElement('div');
    div.innerText = "Click here for focus!";
    div.tabIndex = 1;
    div.contentEditable = "true";
    div.spellcheck = false;
    body.appendChild(div);
    div.addEventListener('click', () => { this.initialize() });
    div.addEventListener('keydown', this.handleKey.bind(this));
    div.addEventListener('keyup', this.handleKey.bind(this));

    const funcDiv = document.createElement('div');
    body.appendChild(funcDiv);
    this.adsFunc = new FuncCanvas(funcDiv);
    this.rFunc = new FuncCanvas(funcDiv);
    this.lowPassFunc = new FuncCanvas(funcDiv);
  }

  initialize() {
    if (this.audioCtx) {
      return;
    }
    this.audioCtx = new AudioContext();

    this.oscilatorNode = new OscillatorNode(
      this.audioCtx, { type: 'triangle' });
    this.oscilatorNode.frequency.setValueAtTime(256, this.audioCtx.currentTime);

    this.lowPassNode = new BiquadFilterNode(this.audioCtx,
      { type: 'lowpass' });
    this.oscilatorNode.connect(this.lowPassNode);

    this.gainNode = new GainNode(this.audioCtx);
    this.lowPassNode.connect(this.gainNode);
    this.gainNode.gain.setValueAtTime(0, this.audioCtx.currentTime);
    this.gainNode.connect(this.audioCtx.destination);

    this.oscilatorNode.start();
  }

  private lastKey = '';
  handleKey(ev: KeyboardEvent) {
    if (ev.type == 'keyup') {
      this.setAmplitude(0);
      this.lastKey = '';
    } else if (ev.type == 'keydown') {
      if (this.lastKey === ev.key) {
        return;
      }
      switch (ev.key) {
        case 'a': this.setNote(0); break;
        case 'w': this.setNote(1); break;
        case 's': this.setNote(2); break;
        case 'e': this.setNote(3); break;
        case 'd': this.setNote(4); break;
        case 'f': this.setNote(5); break;
        case 't': this.setNote(6); break;
        case 'g': this.setNote(7); break;
        case 'y': this.setNote(8); break;
        case 'h': this.setNote(9); break;
        case 'u': this.setNote(10); break;
        case 'j': this.setNote(11); break;
        case 'k': this.setNote(12); break;
      }
      this.setAmplitude(1.0);
      this.lastKey = ev.key;
    }
  }

  scaleCurve(curve: Float32Array | number[], scale: number) {
    const result: number[] = [];
    for (const v of curve) {
      result.push(v * scale);
    }
    return result;
  }
  offsetCurve(curve: Float32Array | number[], offset: number) {
    const result: number[] = [];
    for (const v of curve) {
      result.push(v + offset);
    }
    return result;
  }

  setAmplitude(a: number) {
    let t = this.audioCtx.currentTime;
    this.gainNode.gain.cancelScheduledValues(t);
    if (a <= 0) {
      const currentLevel = this.gainNode.gain.value;
      const decay = this.scaleCurve(this.rFunc.func, currentLevel);
      this.gainNode.gain.setValueCurveAtTime(decay, t, 2.0);
      this.gainNode.gain.setValueAtTime(0, 2.01);
    } else {
      this.lowPassNode.frequency.cancelScheduledValues(t);
      this.gainNode.gain.setValueCurveAtTime(this.adsFunc.func, t, 2.0);
      const currentFreq = this.oscilatorNode.frequency.value;
      const cutoff =
        this.offsetCurve(
          this.scaleCurve(
            this.lowPassFunc.func, currentFreq * 32), currentFreq);
      this.lowPassNode.frequency.setValueCurveAtTime(cutoff, t, 2.0);
    }
  }

  setNote(n: number) {
    const midiNote = n + 60; // Middle C
    const twr2 = Math.pow(2, 1 / 12);
    const freq = 440.0 * Math.pow(twr2, midiNote - 69);
    this.oscilatorNode.frequency.setValueAtTime(
      freq, this.audioCtx.currentTime);
  }

}