package com.alexkbit.deeplearning.digits.rest;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;

import javax.imageio.ImageIO;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.alexkbit.deeplearning.digits.dto.ExecuteResponse;
import com.alexkbit.deeplearning.digits.service.NeuralNetService;

import sun.misc.BASE64Decoder;

@RestController
public class ExecuteController {

    @Autowired
    private NeuralNetService service;

    @RequestMapping(value = "/execute", method = RequestMethod.POST)
    public ExecuteResponse uploadingImage(@RequestParam(value="imageBase64") String imageValue) {
        BufferedImage image = decodeToImage(imageValue.replace("data:image/png;base64,", ""));
        return service.execute(image);
    }

    private static BufferedImage decodeToImage(String imageString) {
        BufferedImage image = null;
        byte[] imageByte;
        try {
            BASE64Decoder decoder = new BASE64Decoder();
            imageByte = decoder.decodeBuffer(imageString);
            ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
            image = ImageIO.read(bis);
            bis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return image;
    }

}
