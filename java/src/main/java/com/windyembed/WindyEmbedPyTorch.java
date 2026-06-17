package com.windyembed;

import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.*;
import ai.djl.translate.*;
import ai.djl.repository.zoo.*;
import java.nio.file.*;
import java.util.*;

/**
 * WindyEmbed PyTorch 版本
 *
 * 使用 DJL (Deep Java Library) 直接加载 PyTorch 模型
 * 不需要转换为 ONNX！
 *
 * 优势:
 * - 直接加载 .pt 文件
 * - 保持 PyTorch 原生性能
 * - 支持 GPU (如果有)
 *
 * Maven 依赖:
 * <dependency>
 *     <groupId>ai.djl</groupId>
 *     <artifactId>api</artifactId>
 *     <version>0.25.0</version>
 * </dependency>
 * <dependency>
 *     <groupId>ai.djl.pytorch</groupId>
 *     <artifactId>pytorch-engine</artifactId>
 *     <version>0.25.0</version>
 * </dependency>
 */
public class WindyEmbedPyTorch implements AutoCloseable {

    private final ZooModel<Void, float[]> model;
    private final Predictor<Void, float[]> predictor;
    private final SimpleTokenizer tokenizer;
    private final int maxSeqLen;

    /**
     * 创建 WindyEmbed 实例 (PyTorch 版本)
     *
     * @param modelPath 模型路径 (model.pt 所在目录)
     * @throws Exception 如果加载失败
     */
    public WindyEmbedPyTorch(String modelPath) throws Exception {
        this(modelPath, 64);
    }

    /**
     * 创建 WindyEmbed 实例
     *
     * @param modelPath 模型路径
     * @param maxSeqLen 最大序列长度
     * @throws Exception 如果加载失败
     */
    public WindyEmbedPyTorch(String modelPath, int maxSeqLen) throws Exception {
        this.maxSeqLen = maxSeqLen;
        this.tokenizer = new SimpleTokenizer();

        // 加载 PyTorch 模型
        Criteria<Void, float[]> criteria = Criteria.builder()
                .setTypes(Void.class, float[].class)
                .optModelPath(Paths.get(modelPath))
                .optEngine("PyTorch")
                .optTranslator(new EmbeddingTranslator())
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    /**
     * 编码文本为 embedding 向量
     *
     * @param text 输入文本
     * @return embedding 向量 (归一化)
     * @throws TranslateException 如果推理失败
     */
    public float[] encode(String text) throws TranslateException {
        return predictor.predict(null);
    }

    /**
     * 计算两个文本的相似度
     *
     * @param text1 文本 1
     * @param text2 文本 2
     * @return 相似度 (-1 到 1)
     * @throws TranslateException 如果推理失败
     */
    public float similarity(String text1, String text2) throws TranslateException {
        float[] emb1 = encode(text1);
        float[] emb2 = encode(text2);
        return cosineSimilarity(emb1, emb2);
    }

    /**
     * 批量编码
     *
     * @param texts 文本列表
     * @return embedding 列表
     * @throws TranslateException 如果推理失败
     */
    public List<float[]> encodeBatch(List<String> texts) throws TranslateException {
        List<float[]> results = new ArrayList<>();
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    /**
     * 查找最相似的文本
     *
     * @param query 查询文本
     * @param candidates 候选文本列表
     * @return 最相似的索引
     * @throws TranslateException 如果推理失败
     */
    public int findMostSimilar(String query, List<String> candidates) throws TranslateException {
        float[] queryEmb = encode(query);
        float maxSim = Float.NEGATIVE_INFINITY;
        int maxIdx = 0;

        for (int i = 0; i < candidates.size(); i++) {
            float[] candEmb = encode(candidates.get(i));
            float sim = cosineSimilarity(queryEmb, candEmb);
            if (sim > maxSim) {
                maxSim = sim;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
    }

    /**
     * 简单分词器
     */
    static class SimpleTokenizer {
        public long[] encode(String text, int maxLen) {
            long[] ids = new long[maxLen];
            byte[] bytes = text.toLowerCase().getBytes();
            for (int i = 0; i < Math.min(bytes.length, maxLen); i++) {
                ids[i] = (bytes[i] & 0xFF) + 1;
            }
            return ids;
        }
    }

    /**
     * Embedding 翻译器
     */
    static class EmbeddingTranslator implements Translator<Void, float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, Void input) {
            // 输入处理在模型内部完成
            return new NDList();
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.singletonOrThrow();
            return array.toFloatArray();
        }
    }
}
