package com.tinymed

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TinyMed — On-Device Chest X-Ray Classifier
 *
 * Runs TFLite inference fully on-device via Android NNAPI delegate.
 * No internet connection required. Inference target: < 50ms cold start.
 *
 * Stage 4 of the TinyMed compression pipeline.
 */
class MainActivity : AppCompatActivity() {

    // ── Constants ──────────────────────────────────────────────────────────
    companion object {
        private const val MODEL_FILE = "tinymed.tflite"
        private const val IMAGE_SIZE = 224
        private const val NUM_CLASSES = 2
        private const val PICK_IMAGE_REQUEST = 1001

        private val CLASS_LABELS = arrayOf("NORMAL", "PNEUMONIA")
        private val CLASS_DESCRIPTIONS = arrayOf(
            "No signs of pneumonia detected.",
            "Potential pneumonia indicators detected. Consult a physician."
        )

        // ImageNet normalization constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    // ── Views ──────────────────────────────────────────────────────────────
    private lateinit var ivXray: ImageView
    private lateinit var btnSelectImage: Button
    private lateinit var btnRunInference: Button
    private lateinit var tvResult: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvDescription: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var cardResult: View
    private lateinit var tvModelInfo: TextView
    private lateinit var tvDisclaimer: TextView

    // ── TFLite ─────────────────────────────────────────────────────────────
    private var interpreter: Interpreter? = null
    private var currentBitmap: Bitmap? = null

    // ─────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupClickListeners()
        loadModel()
        updateModelInfo()
    }

    // ── View Binding ───────────────────────────────────────────────────────
    private fun bindViews() {
        ivXray         = findViewById(R.id.iv_xray)
        btnSelectImage = findViewById(R.id.btn_select_image)
        btnRunInference = findViewById(R.id.btn_run_inference)
        tvResult       = findViewById(R.id.tv_result)
        tvConfidence   = findViewById(R.id.tv_confidence)
        tvLatency      = findViewById(R.id.tv_latency)
        tvDescription  = findViewById(R.id.tv_description)
        progressBar    = findViewById(R.id.progress_bar)
        cardResult     = findViewById(R.id.card_result)
        tvModelInfo    = findViewById(R.id.tv_model_info)
        tvDisclaimer   = findViewById(R.id.tv_disclaimer)

        cardResult.visibility = View.GONE
        btnRunInference.isEnabled = false
    }

    // ── Click Listeners ────────────────────────────────────────────────────
    private fun setupClickListeners() {
        btnSelectImage.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            intent.type = "image/*"
            startActivityForResult(intent, PICK_IMAGE_REQUEST)
        }

        btnRunInference.setOnClickListener {
            currentBitmap?.let { runInference(it) }
                ?: showToast("Please select a chest X-ray image first.")
        }
    }

    // ── Model Loading ──────────────────────────────────────────────────────
    private fun loadModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, MODEL_FILE)
            val options = Interpreter.Options().apply {
                numThreads = 4
                // Enable NNAPI delegate for Android NPU acceleration
                useNNAPI = true
                // Fallback to GPU if NNAPI unavailable
                useXNNPACK = true
            }
            interpreter = Interpreter(modelBuffer, options)
            showToast("Model loaded. Using Android NNAPI.")
        } catch (e: IOException) {
            showToast("Model load failed: ${e.message}")
        }
    }

    // ── Image Selection ────────────────────────────────────────────────────
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK) {
            val uri: Uri = data?.data ?: return
            try {
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                currentBitmap = bitmap
                ivXray.setImageBitmap(bitmap)
                btnRunInference.isEnabled = true
                cardResult.visibility = View.GONE
                tvDisclaimer.visibility = View.VISIBLE
            } catch (e: IOException) {
                showToast("Failed to load image: ${e.message}")
            }
        }
    }

    // ── Preprocessing ──────────────────────────────────────────────────────
    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16 and 0xFF) / 255.0f - MEAN[0]) / STD[0]
            val g = ((pixel shr 8  and 0xFF) / 255.0f - MEAN[1]) / STD[1]
            val b = ((pixel        and 0xFF) / 255.0f - MEAN[2]) / STD[2]
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    // ── Inference ──────────────────────────────────────────────────────────
    private fun runInference(bitmap: Bitmap) {
        val tflite = interpreter ?: run {
            showToast("Model not loaded.")
            return
        }

        progressBar.visibility = View.VISIBLE
        btnRunInference.isEnabled = false
        cardResult.visibility = View.GONE

        Thread {
            try {
                val inputBuffer = preprocessBitmap(bitmap)
                val outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }

                // ── Timed inference ──
                val startNs = System.nanoTime()
                tflite.run(inputBuffer, outputBuffer)
                val latencyMs = (System.nanoTime() - startNs) / 1_000_000.0

                val logits = outputBuffer[0]
                val maxLogit = logits.max()
                val expLogits = logits.map { Math.exp((it - maxLogit).toDouble()) }
                val sumExp = expLogits.sum()
                val probs = expLogits.map { (it / sumExp).toFloat() }

                val predictedClass = probs.indices.maxByOrNull { probs[it] } ?: 0
                val confidence = probs[predictedClass] * 100.0f

                runOnUiThread {
                    displayResult(predictedClass, confidence, latencyMs)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    showToast("Inference error: ${e.message}")
                    progressBar.visibility = View.GONE
                    btnRunInference.isEnabled = true
                }
            }
        }.start()
    }

    // ── Results Display ────────────────────────────────────────────────────
    private fun displayResult(classIdx: Int, confidence: Float, latencyMs: Double) {
        progressBar.visibility = View.GONE
        btnRunInference.isEnabled = true
        cardResult.visibility = View.VISIBLE

        tvResult.text = CLASS_LABELS[classIdx]
        tvConfidence.text = "Confidence: ${String.format("%.1f", confidence)}%"
        tvLatency.text = "Inference: ${String.format("%.1f", latencyMs)}ms on-device"
        tvDescription.text = CLASS_DESCRIPTIONS[classIdx]

        // Color-code result
        val colorRes = if (classIdx == 0)
            android.R.color.holo_green_dark
        else
            android.R.color.holo_red_dark
        tvResult.setTextColor(getColor(colorRes))
    }

    // ── Model Info ─────────────────────────────────────────────────────────
    private fun updateModelInfo() {
        val info = buildString {
            append("Model: EfficientNet-B0 (Knowledge Distilled)\n")
            append("Runtime: TFLite + NNAPI delegate\n")
            append("Input: 224×224 RGB (ImageNet normalized)\n")
            append("Classes: Normal / Pneumonia\n")
            append("Compression: ResNet-18 → EfficientNet-B0 via KD\n")
            append("Target size: <5MB | Target latency: <50ms")
        }
        tvModelInfo.text = info
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    private fun showToast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}
