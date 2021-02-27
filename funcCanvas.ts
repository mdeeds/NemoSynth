export class FuncCanvas {
  canvas: HTMLCanvasElement;
  func: Float32Array;
  constructor(container: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.classList.add('funcCanvas');
    this.canvas.width = 100;
    this.canvas.height = 100;
    this.canvas.tabIndex = 1000;
    this.canvas.addEventListener('mousemove', this.handleMouse.bind(this));
    container.appendChild(this.canvas);
    this.func = new Float32Array(100);
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  handleMouse(ev: MouseEvent) {
    if (ev.buttons > 0) {
      const v = (100 - ev.offsetY) / 100.0;
      const i = Math.round(ev.offsetX);
      this.func[i] = v;
    }
  }

  renderLoop() {
    const ctx = this.canvas.getContext('2d');
    ctx.fillStyle = '#f29';
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (let i = 0; i < this.func.length; ++i) {
      const v = this.func[i];
      ctx.fillRect(i, 99 - v * 100, 1, 1);
    }
    requestAnimationFrame(this.renderLoop.bind(this));
  }
}