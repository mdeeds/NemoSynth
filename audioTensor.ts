import { Tensor } from "@tensorflow/tfjs";
import * as tf from '@tensorflow/tfjs';

export class AudioTensor {
  static readonly audioCtx: AudioContext = new AudioContext();

  static loadClip(url: string): Promise<AudioBuffer> {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open('GET', url, true);
      request.responseType = 'arraybuffer';
      request.onload = () => {
        AudioTensor.audioCtx.decodeAudioData(request.response, (buffer) => {
          resolve(buffer);
        }, reject);
      }
      request.send();
    });
  }

  static getTensor(buffer: AudioBuffer, sampleOffset: number, sampleCount: number) {
    const rawData = new Float32Array(sampleCount);
    buffer.copyFromChannel(rawData, 0, sampleOffset);
    const result: Tensor = tf.tensor(rawData, [sampleCount], "float32");
    return result;
  }
}