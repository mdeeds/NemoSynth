import * as tf from '@tensorflow/tfjs';
import { Log } from './log';
import { MemoryTest } from './memoryTest';
import { SmokeTest } from './smokeTest';
import { Synth } from './synth';
import { VGG } from './vgg';


// The following statistics are for 44,800 samples
// Just building a single embedding.  (E.g. half the cost function.)
// Backend: webgl
// Elapsed: 56945.57000003988ms
// WebGL Memory: 
// "numBytesInGPU":86,215,512,
// "numBytesInGPUAllocated":111,570,744,
// "numBytesInGPUFree":25,355,232,
// "numTensors":240,055,
// "numDataBuffers":192,051,
// "numBytes":49,538,132}

// Backend: cpu
// Elapsed: 2759ms
// "numTensors":240,055,
// "numDataBuffers":192,045,
// "numBytes":49,537,940}

//tf.setBackend('cpu');
//SmokeTest.smoke();



//tf.setBackend('webgl');
// tf.setBackend('cpu');
//const vgg = new VGG();

// tf.setBackend('webgl');
// Log.info("Start Test");
// MemoryTest.simpleAdd();
// MemoryTest.detachLoop2();
// MemoryTest.assignLoop();
// MemoryTest.modelLoop();
// Log.info("End Test");

function main() {
  const s = new Synth();
}

setTimeout(main);