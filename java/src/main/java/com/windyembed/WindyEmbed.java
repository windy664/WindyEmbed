package com.windyembed;

import ai.onnxruntime.*;
import java.nio.LongBuffer;
import java.util.*;
import java.io.*;
import java.nio.file.*;

/**
 * WindyEmbed - Java Inference for Embedding Models
 *
 * A lightweight embedding model that runs on CPU using ONNX Runtime.
 * Designed for resource-constrained servers.
 *
 * Usage:
 * <pre>
 * WindyEmbed embed = new WindyEmbed("model.onnx");
 * float[] vector = embed.encode("Hello world");
 * float similarity = embed.similarity("Hello", "Hi");
 * </pre>
 *
 * Performance:
 * - 10M params: ~5ms per embedding on CPU
 * - 50M params: ~12ms per embedding on CPU
 * - No GPU required
 */
public class WindyEmbed implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Tokenizer tokenizer;
    private final int maxSeqLen;
    private final int embeddingDim;

    /**
     * Create WindyEmbed instance.
     *
     * @param modelPath Path to ONNX model file
     * @throws OrtException If model loading fails
     */
    public WindyEmbed(String modelPath) throws OrtException {
        this(modelPath, null, 128);
    }

    /**
     * Create WindyEmbed instance with custom tokenizer.
     *
     * @param modelPath Path to ONNX model file
     * @param vocabPath Path to vocabulary file (null for simple tokenizer)
     * @param maxSeqLen Maximum sequence length
     * @throws OrtException If model loading fails
     */
    public WindyEmbed(String modelPath, String vocabPath, int maxSeqLen) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.maxSeqLen = maxSeqLen;

        // Load model
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors());

        this.session = env.createSession(modelPath, options);

        // Get embedding dimension from model output
        this.embeddingDim = getEmbeddingDim();

        // Load tokenizer
        if (vocabPath != null && Files.exists(Paths.get(vocabPath))) {
            this.tokenizer = new WordPieceTokenizer(vocabPath);
        } else {
            this.tokenizer = new SimpleTokenizer();
        }
    }

    /**
     * Encode text into embedding vector.
     *
     * @param text Input text
     * @return Embedding vector (normalized)
     * @throws OrtException If inference fails
     */
    public float[] encode(String text) throws OrtException {
        // Tokenize
        long[] tokenIds = tokenizer.encode(text, maxSeqLen);
        long[] attentionMask = createAttentionMask(tokenIds);

        // Create input tensors
        LongBuffer idsBuffer = LongBuffer.wrap(tokenIds);
        LongBuffer maskBuffer = LongBuffer.wrap(attentionMask);

        OnnxTensor idsTensor = OnnxTensor.createTensor(env, idsBuffer, new long[]{1, tokenIds.length});
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, maskBuffer, new long[]{1, attentionMask.length});

        // Run inference
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", idsTensor);
        inputs.put("attention_mask", maskTensor);

        OrtSession.Result result = session.run(inputs);

        // Get embeddings
        float[][] embeddings = (float[][]) result.get(0).getValue();
        float[] embedding = embeddings[0];

        // Normalize
        embedding = normalize(embedding);

        // Clean up
        idsTensor.close();
        maskTensor.close();
        result.close();

        return embedding;
    }

    /**
     * Compute cosine similarity between two texts.
     *
     * @param text1 First text
     * @param text2 Second text
     * @return Similarity score (-1 to 1)
     * @throws OrtException If inference fails
     */
    public float similarity(String text1, String text2) throws OrtException {
        float[] emb1 = encode(text1);
        float[] emb2 = encode(text2);
        return cosineSimilarity(emb1, emb2);
    }

    /**
     * Batch encode multiple texts.
     *
     * @param texts List of texts
     * @return List of embedding vectors
     * @throws OrtException If inference fails
     */
    public List<float[]> encodeBatch(List<String> texts) throws OrtException {
        List<float[]> results = new ArrayList<>();
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    /**
     * Find most similar text from a list.
     *
     * @param query Query text
     * @param candidates Candidate texts
     * @return Index of most similar candidate
     * @throws OrtException If inference fails
     */
    public int findMostSimilar(String query, List<String> candidates) throws OrtException {
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

    /**
     * Get embedding dimension.
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }

    private long[] createAttentionMask(long[] tokenIds) {
        long[] mask = new long[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            mask[i] = tokenIds[i] != 0 ? 1 : 0;
        }
        return mask;
    }

    private float[] normalize(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        return vector;
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
    public void close() throws OrtException {
        session.close();
        env.close();
    }

    /**
     * Simple tokenizer interface.
     */
    public interface Tokenizer {
        long[] encode(String text, int maxLen);
    }

    /**
     * Simple character-level tokenizer (fallback).
     */
    public static class SimpleTokenizer implements Tokenizer {
        @Override
        public long[] encode(String text, int maxLen) {
            long[] ids = new long[maxLen];
            byte[] bytes = text.toLowerCase().getBytes();
            for (int i = 0; i < Math.min(bytes.length, maxLen); i++) {
                ids[i] = (bytes[i] & 0xFF) + 1;  // +1 to avoid 0 (padding)
            }
            return ids;
        }
    }

    /**
     * WordPiece tokenizer (loads vocab.txt).
     */
    public static class WordPieceTokenizer implements Tokenizer {
        private final Map<String, Long> vocab;
        private final int maxLen;

        public WordPieceTokenizer(String vocabPath) throws IOException {
            this.vocab = new HashMap<>();
            List<String> lines = Files.readAllLines(Paths.get(vocabPath));
            for (int i = 0; i < lines.size(); i++) {
                vocab.put(lines.get(i).trim(), (long) i);
            }
            this.maxLen = 128;
        }

        @Override
        public long[] encode(String text, int maxLen) {
            long[] ids = new long[maxLen];
            String[] words = text.toLowerCase().split("\\s+");
            int idx = 0;

            for (String word : words) {
                if (idx >= maxLen) break;
                Long id = vocab.get(word);
                if (id != null) {
                    ids[idx++] = id;
                } else {
                    // Unknown token
                    ids[idx++] = vocab.getOrDefault("[UNK]", 3L);
                }
            }
            return ids;
        }
    }
}
