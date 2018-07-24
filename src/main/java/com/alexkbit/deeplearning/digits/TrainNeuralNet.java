package com.alexkbit.deeplearning.digits;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainNeuralNet {

    private static String NN_MODEL_NAME = "nnModel";
    private static String RESOURCE_PATH = "./src/main/resources/";

    private static final int BATCH_SIZE = 242;

    private static final int IMG_SIZE = 50;
    private static final int INPUT_COUNT = IMG_SIZE * IMG_SIZE;
    private static final int OUTPUT_COUNT = 10;
    private static final int MEDIUM_COUNT = INPUT_COUNT / OUTPUT_COUNT;

    public static void main(String[] args) throws IOException, InterruptedException {

        DataSet allData;
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(Paths.get(DataSetPrepare.DATA_SET_FILE).toFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, INPUT_COUNT, OUTPUT_COUNT);
            allData = iterator.next();
        }

        allData.shuffle(42);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        System.out.println("All data ready");
        MultiLayerNetwork model = new MultiLayerNetwork(netConfig());
        model.init();
        model.fit(trainingData);
        System.out.println("Fit done");

        INDArray output = model.output(testData.getFeatureMatrix());
        System.out.println("Output: " + output);

        Evaluation eval = new Evaluation(OUTPUT_COUNT);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

        ModelSerializer.writeModel(model, new File(RESOURCE_PATH + NN_MODEL_NAME), false);
        System.out.println("Model Saved");

    }

    private static MultiLayerConfiguration netConfig() {
        return new NeuralNetConfiguration.Builder()
                .iterations(1000)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true)
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(INPUT_COUNT)
                        .nOut(MEDIUM_COUNT)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(MEDIUM_COUNT)
                        .nOut(MEDIUM_COUNT)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(MEDIUM_COUNT)
                        .nOut(OUTPUT_COUNT)
                        .build())
                .backprop(true).pretrain(false)
                .build();
    }


}
