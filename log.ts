import { info } from "console";

export class Log {
  private static div: HTMLDivElement = Log.initialDiv();
  static info(message: any) {
    const div = document.createElement('div');
    console.log(message);

    if (typeof (message) === "string") {
      div.innerText =
        `${(window.performance.now() / 1000).toFixed(2)}: ${message}`;
    } else {
      div.innerText = JSON.stringify(message, null, 1);
      div.style.setProperty('color', 'blue');
    }
    Log.div.appendChild(div);
    while (Log.div.children.length > 15) {
      Log.div.children[0].remove();
    }
  }

  static initialDiv() {
    const div = document.createElement('div');
    setTimeout(() => {
      console.log("Adding logging...");
      const body = document.getElementsByTagName('body')[0];
      body.appendChild(div);
    }, 100);
    return div;
  }
}