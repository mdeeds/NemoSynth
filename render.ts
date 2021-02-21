import * as tf from '@tensorflow/tfjs';
import { Tensor, TypedArray } from "@tensorflow/tfjs";

export class Render {
  static message(text: string) {
    const body = document.getElementsByTagName('body')[0];
    const div = document.createElement('div');
    div.innerText = text;
    body.appendChild(div);
  }

  static addCanvases(t: Tensor) {
    if (t.shape.length === 2) {
      const [batchSize, n] = t.shape;
      Render.addCanvases(tf.reshape(t, [batchSize, n, 1]));
    } else if (t.shape.length === 3) {
      const [batchSize, width, height] = t.shape;
      const lastBatch = Math.min(10, batchSize);
      const tensors = tf.split(t, batchSize, 0);
      const body = document.getElementsByTagName('body')[0];
      for (let i = 0; i < batchSize; ++i) {
        const imageTensor = tf.reshape(tensors[i], [width, height]);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        body.appendChild(canvas);
        imageTensor.data()
          .then((data) => {
            Render.drawCanvas(data, canvas);
          })
      }
    } else {
      console.error(`Unsupported shape: ${t.shape}`);
    }
  }


  static drawCanvas(data: TypedArray, canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    let minC = 255;
    let maxC = 0;
    ctx.beginPath();

    const imageData = ctx.createImageData(canvas.width, canvas.height);

    for (let x = 0; x < canvas.width; ++x) {
      for (let y = 0; y < canvas.height; ++y) {
        let v = data[y + x * canvas.height] * 128 * 10 + 128;
        v = Math.round(v);
        v = Math.min(255, Math.max(0, v))
        minC = Math.min(minC, v);
        maxC = Math.max(maxC, v);
        let i = (x + y * canvas.width) * 4;
        let rgb = [0, 0, 0];
        if (v > 128) {
          rgb = [2 * (v - 128), 64, 64];
        }
        if (v < 128) {
          rgb = [64, 64, 2 * (128 - v)];
        }
        imageData.data[i + 0] = rgb[0];
        imageData.data[i + 1] = rgb[1];
        imageData.data[i + 2] = rgb[2];
        imageData.data[i + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

}