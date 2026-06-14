# TinyMed — ProGuard / R8 rules
#
# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep TensorFlow Lite GPU delegate classes (loaded via reflection)
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# Keep TensorFlow Lite support library classes used for image/tensor processing
-keep class org.tensorflow.lite.support.** { *; }
-dontwarn org.tensorflow.lite.support.**

# Keep TensorFlow Lite core interpreter classes
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
