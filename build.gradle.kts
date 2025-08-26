import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension

plugins {
  kotlin("multiplatform") version "2.2.10"
  id("com.android.library") version "8.11.1"
  id("org.jetbrains.kotlin.android") version "2.2.10" apply false
  id("maven-publish")
}

group = "ltd.mbor.sciko"
version = "0.1-SNAPSHOT"

repositories {
  mavenCentral()
  google()
  maven {
    name = "GitHubPackages"
    url = uri("https://maven.pkg.github.com/mihbor/sciko-linalg")
    credentials {
      username = project.findProperty("gpr.user") as String? ?: System.getenv("USERNAME")
      password = project.findProperty("gpr.key") as String? ?: System.getenv("TOKEN")
    }
  }
}

kotlin {
  jvm()
  jvmToolchain(21)
  js(IR) {
    browser()
  }
  androidTarget {
    publishLibraryVariants("release", "debug")
  }
  sourceSets {
    val commonMain by getting {
      dependencies {
        api("org.jetbrains.kotlinx:multik-core:0.2.3")
        implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
        implementation("com.ionspin.kotlin:bignum:0.3.10")
        api("ltd.mbor.sciko:sciko-linalg:0.1-SNAPSHOT")
      }
    }
    val commonTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
    val jvmTest by getting {
      dependencies {
        implementation(kotlin("test"))
        implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.8.0")
      }
    }
  }
}

android {
  compileSdk = 36
  namespace = "ltd.mbor.sciko.robotics"
  defaultConfig {
    minSdk = 21
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
  }
}

publishing {
  repositories {
    maven {
      name = "GitHubPackages"
      url = uri("https://maven.pkg.github.com/mihbor/sciko-robotics")
      credentials {
        username = project.findProperty("gpr.user") as String? ?: System.getenv("USERNAME")
        password = project.findProperty("gpr.key") as String? ?: System.getenv("TOKEN")
      }
    }
  }
}

rootProject.plugins.withType(YarnPlugin::class.java) {
  rootProject.the<YarnRootExtension>().yarnLockAutoReplace = true
}
