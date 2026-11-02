package com.example.fruitripeness

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.view.View
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.card.MaterialCardView
import com.google.android.material.progressindicator.LinearProgressIndicator
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    private lateinit var imageView: ImageView
    private lateinit var placeholder: TextView
    private lateinit var loading: ProgressBar
    private lateinit var resultCard: MaterialCardView
    private lateinit var resultTitle: TextView
    private lateinit var resultConfidence: TextView
    private lateinit var resultAdvice: TextView
    private lateinit var resultTime: TextView
    private lateinit var names: List<TextView>
    private lateinit var bars: List<LinearProgressIndicator>

    private var photoUri: Uri? = null

    companion object {
        private const val IMG_SIZE = 224
        private const val MAX_DISPLAY = 1024      // taille max de l'image décodée
        private const val CONFIDENCE_MIN = 0.70f  // en dessous : "Incertain"

        // Conseil affiché pour chaque classe
        private val ADVICE = mapOf(
            "banane_non_mure" to "Encore verte : laissez-la mûrir 3 à 5 jours à température ambiante.",
            "banane_mure" to "Parfaite : à consommer dans les 1 à 2 jours.",
            "banane_trop_mure" to "Très mûre : idéale pour un gâteau, des pancakes ou un smoothie.",
            "fraise_non_mure" to "Pas encore prête : une fraise ne mûrit plus après la cueillette, attendez qu'elle rougisse sur le plant.",
            "fraise_partiellement_mure" to "Presque prête : elle sera plus sucrée dans 1 à 2 jours.",
            "fraise_mure" to "Prête : à déguster rapidement, de préférence dans les 2 jours.",
            "fraise_pourrie" to "Abîmée ou moisie : ne la mangez pas, jetez-la.",
            "avocat_non_mur" to "Encore dur : gardez-le à température ambiante 3 à 5 jours.",
            "avocat_partiellement_mur" to "Bientôt prêt : encore 1 à 2 jours.",
            "avocat_mur" to "À point : à consommer dans les 1 à 2 jours.",
            "avocat_pourri" to "Trop mûr ou abîmé : vérifiez la chair avant de le manger."
        )
    }

    // Galerie (aucune permission nécessaire)
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { process(it) }
    }

    // Appareil photo en pleine résolution
    private val takePhoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) photoUri?.let { process(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(b.left, b.top, b.right, b.bottom)
            insets
        }

        imageView = findViewById(R.id.imageView)
        placeholder = findViewById(R.id.placeholder)
        loading = findViewById(R.id.loading)
        resultCard = findViewById(R.id.resultCard)
        resultTitle = findViewById(R.id.resultTitle)
        resultConfidence = findViewById(R.id.resultConfidence)
        resultAdvice = findViewById(R.id.resultAdvice)
        resultTime = findViewById(R.id.resultTime)
        names = listOf(R.id.name1, R.id.name2, R.id.name3).map { findViewById<TextView>(it) }
        bars = listOf(R.id.bar1, R.id.bar2, R.id.bar3).map { findViewById<LinearProgressIndicator>(it) }

        interpreter = Interpreter(loadModel("fruit_ripeness.tflite"))
        labels = assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

        findViewById<View>(R.id.btnGallery).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<View>(R.id.btnCamera).setOnClickListener {
            val dir = File(cacheDir, "photos").apply { mkdirs() }
            val file = File(dir, "photo.jpg")
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            photoUri = uri
            takePhoto.launch(uri)
        }
    }

    private fun loadModel(name: String): ByteBuffer {
        val bytes = assets.open(name).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes); rewind()
        }
    }

    /** Décodage + inférence hors du thread principal, affichage ensuite. */
    private fun process(uri: Uri) {
        loading.visibility = View.VISIBLE
        resultCard.visibility = View.GONE
        thread {
            try {
                val bitmap = loadBitmap(uri)
                val t0 = SystemClock.elapsedRealtime()
                val probs = classify(bitmap)
                val ms = SystemClock.elapsedRealtime() - t0
                runOnUiThread { show(bitmap, probs, ms) }
            } catch (e: Exception) {
                runOnUiThread {
                    loading.visibility = View.GONE
                    resultCard.visibility = View.VISIBLE
                    resultTitle.text = "Erreur"
                    resultConfidence.text = e.message ?: "Image illisible"
                }
            }
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap =
        if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE   // gère aussi l'orientation EXIF
                val longest = maxOf(info.size.width, info.size.height)
                if (longest > MAX_DISPLAY) {
                    val s = MAX_DISPLAY.toFloat() / longest
                    decoder.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
                }
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }

    /** Retourne les probabilités des 11 classes. */
    private fun classify(bitmap: Bitmap): FloatArray {
        // 1) Redimensionner en 224x224 (comme à l'entraînement)
        val scaled = Bitmap.createScaledBitmap(
            bitmap.copy(Bitmap.Config.ARGB_8888, false), IMG_SIZE, IMG_SIZE, true)

        // 2) Pixels RGB float32 entre 0 et 255 (la normalisation est dans le modèle)
        val input = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        scaled.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())  // R
            input.putFloat(((p shr 8) and 0xFF).toFloat())   // G
            input.putFloat((p and 0xFF).toFloat())           // B
        }
        input.rewind()

        // 3) Inférence
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        return output[0]
    }

    private fun show(bitmap: Bitmap, probs: FloatArray, ms: Long) {
        loading.visibility = View.GONE
        placeholder.visibility = View.GONE
        imageView.setImageBitmap(bitmap)

        val top = probs.withIndex().sortedByDescending { it.value }.take(3)
        val best = labels[top[0].index]
        val conf = top[0].value
        val uncertain = conf < CONFIDENCE_MIN
        val color = stageColor(best)

        resultCard.visibility = View.VISIBLE
        resultCard.strokeColor = if (uncertain) Color.parseColor("#9E9E9E") else color

        if (uncertain) {
            resultTitle.text = "❓ Incertain"
            resultTitle.setTextColor(Color.parseColor("#757575"))
            resultConfidence.text = "Meilleure hypothèse : ${pretty(best)} (%.0f %%)".format(conf * 100)
            resultAdvice.text = "Le modèle hésite. Essayez avec un seul fruit, bien cadré, " +
                "sur un fond uni et avec une bonne lumière."
        } else {
            resultTitle.text = pretty(best)
            resultTitle.setTextColor(color)
            resultConfidence.text = "Confiance : %.1f %%".format(conf * 100)
            resultAdvice.text = "💡 " + (ADVICE[best] ?: "")
        }

        for (i in 0..2) {
            val label = labels[top[i].index]
            val c = if (i == 0 && !uncertain) color else Color.parseColor("#9E9E9E")
            names[i].text = "%s  —  %.1f %%".format(pretty(label), top[i].value * 100)
            bars[i].setIndicatorColor(c)
            bars[i].setProgressCompat((top[i].value * 100).toInt(), true)
        }
        resultTime.text = "Inférence : $ms ms · exécutée sur le téléphone, sans Internet"
    }

    /** Vert = pas mûr, orange = mûr, rouge/brun = abîmé. */
    private fun stageColor(label: String): Int = when {
        label.endsWith("non_mure") || label.endsWith("non_mur") -> Color.parseColor("#2E7D32")
        label.contains("partiellement") -> Color.parseColor("#F9A825")
        label.endsWith("pourri") || label.endsWith("pourrie") || label.endsWith("trop_mure") ->
            Color.parseColor("#C62828")
        else -> Color.parseColor("#EF6C00")   // mûr
    }

    private fun pretty(label: String) =
        label.replace("_", " ").replaceFirstChar { it.uppercase() }
            .replace("mure", "mûre").replace("mur", "mûr")

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
