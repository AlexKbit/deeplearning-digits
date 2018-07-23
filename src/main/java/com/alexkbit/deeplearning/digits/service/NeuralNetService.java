package com.alexkbit.deeplearning.digits.service;

import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.annotation.PostConstruct;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import com.alexkbit.deeplearning.digits.dto.ExecuteResponse;
import com.alexkbit.deeplearning.digits.util.ConvertUtil;

@Service
public class NeuralNetService {

    @Value(value = "classpath:nnModel")
    private Resource nnModel;

    private MultiLayerNetwork model;

    @PostConstruct
    public void init() throws IOException {
        model = ModelSerializer.restoreMultiLayerNetwork(nnModel.getInputStream());
    }

    public ExecuteResponse execute(BufferedImage image) {
        INDArray array = ConvertUtil.imageToNdArray(image);
        INDArray result = model.output(array);
        return defineDigit(result);
    }

    private static ExecuteResponse defineDigit(INDArray result) {
        ExecuteResponse response = new ExecuteResponse();
        double max = -1;
        int index = -1;
        for (int i=0; i<result.length(); i++) {
            double v = result.getDouble(i);
            response.addProbability(v);
            if (v > max) {
                max = v;
                index = i;
            }
        }
        response.setAnswer(index);
        response.setEps(max);
        return response;
    }

}
