package com.alexkbit.deeplearning.digits.util;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ConvertUtil {

    public static double[] imageToTensor(BufferedImage image) {
        double[] tensor = new double[image.getHeight() * image.getWidth()];
        for (int i=0; i<image.getWidth(); i++) {
            for (int j=0; j<image.getHeight(); j++) {
                tensor[j*image.getWidth()+i]=isAlpha(image.getRGB(i,j)) ? 1 : 0;
            }
        }
        return tensor;
    }

    public static int[] convertToTensor(Path file) throws IOException {
        BufferedImage image = ImageIO.read(file.toFile());
        int[] tensor = new int[image.getHeight() * image.getWidth()];
        for (int i=0; i<image.getWidth(); i++) {
            for (int j=0; j<image.getHeight(); j++) {
                tensor[j*image.getWidth()+i]=isAlpha(image.getRGB(i,j)) ? 1 : 0;
            }
        }
        return tensor;
    }

    public static INDArray imageToNdArray(BufferedImage image) {
        return Nd4j.create(imageToTensor(image));
    }

    private static boolean isAlpha(int rgb)
    {
        return (rgb & 0xFF000000) == 0xFF000000;
    }
}
