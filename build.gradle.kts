plugins {
  kotlin("multiplatform") version "2.0.21"
  id("maven-publish")
}

group = "com.github.mihbor"
version = "0.1-SNAPSHOT"

repositories {
  mavenCentral()
  maven("https://jitpack.io")
}

kotlin {
  sourceSets {
    val commonMain by getting {
      dependencies {
        api("org.jetbrains.kotlinx:multik-core:0.2.3")
        implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
        implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.7.0")
        implementation("com.ionspin.kotlin:bignum:0.3.10")
        implementation("com.github.mihbor:sciko-linalg:1edefdf")
      }
    }
    val commonTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
  }
  jvm {
    testRuns["test"].executionTask.configure {
      useJUnitPlatform()
    }
  }
  jvmToolchain(21)
}
