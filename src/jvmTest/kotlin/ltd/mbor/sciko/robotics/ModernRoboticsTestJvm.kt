package ltd.mbor.sciko.robotics

import ltd.mbor.sciko.linalg.diagonal
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.File
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals

class ModernRoboticsTestJvm {

  @Test
  fun `test InverseDynamicsTrajectory`() {

    // Create a trajectory to follow using functions from Chapter 9
    val thetastart = mk.ndarray(mk[0, 0, 0]).map{ it.toDouble() }
    val thetaend = mk.ndarray(mk[PI/2, PI/2, PI/2])
    val Tf = 3.0
    val N = 1000
    val method = 5
    val traj = JointTrajectory(thetastart, thetaend, Tf, N, method)
    val thetamat = traj
    val dthetamat = mk.zeros<Double>(1000, 3)
    val ddthetamat = mk.zeros<Double>(1000, 3)
    val dt = Tf / (N - 1.0)
    for (i in (0..<traj.shape[0] - 1)) {
      dthetamat[i + 1] = (thetamat[i + 1]-thetamat[i]) / dt
      ddthetamat[i + 1] = (dthetamat[i + 1] - dthetamat[i]) / dt
    }

    // Initialize robot description (Example with 3 links)
    val taumat = InverseDynamicsTrajectory(
      thetamat = thetamat,
      dthetamat = dthetamat,
      ddthetamat = ddthetamat,
      g = mk.ndarray(mk[0.0, 0.0, -9.8]),
      Ftipmat = mk.ones(N, 6),
      Mlist = mk.ndarray(mk[
        mk[
          mk[1.0, 0.0, 0.0, 0.0     ],
          mk[0.0, 1.0, 0.0, 0.0     ],
          mk[0.0, 0.0, 1.0, 0.089159],
          mk[0.0, 0.0, 0.0, 1.0     ]
        ],
        mk[
          mk[ 0.0, 0.0, 1.0, 0.28   ],
          mk[ 0.0, 1.0, 0.0, 0.13585],
          mk[-1.0, 0.0, 0.0, 0.0    ],
          mk[ 0.0, 0.0, 0.0, 1.0    ]
        ],
        mk[
          mk[1.0, 0.0, 0.0,  0.0   ],
          mk[0.0, 1.0, 0.0, -0.1197],
          mk[0.0, 0.0, 1.0,  0.395 ],
          mk[0.0, 0.0, 0.0,  1.0   ]
        ],
        mk[
          mk[1.0, 0.0, 0.0, 0.0    ],
          mk[0.0, 1.0, 0.0, 0.0    ],
          mk[0.0, 0.0, 1.0, 0.14225],
          mk[0.0, 0.0, 0.0, 1.0    ]
        ]
      ]),
      Glist = mk.diagonal(mk[0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275]).reshape(1, 6, 6),
      Slist = mk.ndarray(mk[
        mk[1.0,  0.0,  1.0,  0.0  ,  1.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.425]
      ]).transpose(),
    )
    assertEquals(
      mk.d2arrayFromFile("InverseDynamicsTrajectory.csv"),
      taumat,
      "$taumat not equal ${mk.d2arrayFromFile("InverseDynamicsTrajectory.csv")}"
    )

    //Output using kandy to plot the joint forces/torques
    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
    plotNVector("InverseDynamicsTrajectory", timestamps, taumat.toListD2() to "Tau")
  }

  @Test
  fun `test ForwardDynamicsTrajectory`() {
    val (thetamat, dthetamat) = ForwardDynamicsTrajectory(
      thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
      dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
      taumat = mk.ndarray(mk[
        mk[3.63, -6.58, -5.57], mk[3.74, -5.55, -5.5 ],
        mk[4.31, -0.68, -5.19], mk[5.18,  5.63, -4.31],
        mk[5.85,  8.17, -2.59], mk[5.78,  2.79, -1.7 ],
        mk[4.99, -5.3 , -1.19], mk[4.08, -9.41,  0.07],
        mk[3.56,-10.1 ,  0.97], mk[3.49, -9.41,  1.23]
      ]),
      g = mk.ndarray(mk[0.0, 0.0, -9.8]),
      Ftipmat = mk.ones<Double>(10, 6),
      Mlist = mk.ndarray(mk[
        mk[
          mk[1.0, 0.0, 0.0, 0.0     ],
          mk[0.0, 1.0, 0.0, 0.0     ],
          mk[0.0, 0.0, 1.0, 0.089159],
          mk[0.0, 0.0, 0.0, 1.0     ]
        ],
        mk[
          mk[ 0.0, 0.0, 1.0, 0.28   ],
          mk[ 0.0, 1.0, 0.0, 0.13585],
          mk[-1.0, 0.0, 0.0, 0.0    ],
          mk[ 0.0, 0.0, 0.0, 1.0    ]
        ],
        mk[
          mk[1.0, 0.0, 0.0,  0.0   ],
          mk[0.0, 1.0, 0.0, -0.1197],
          mk[0.0, 0.0, 1.0,  0.395 ],
          mk[0.0, 0.0, 0.0,  1.0   ]
        ],
        mk[
          mk[1.0, 0.0, 0.0, 0.0    ],
          mk[0.0, 1.0, 0.0, 0.0    ],
          mk[0.0, 0.0, 1.0, 0.14225],
          mk[0.0, 0.0, 0.0, 1.0    ]
        ]
      ]),
      Glist = mk.diagonal(mk[0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275]).reshape(1, 6, 6),
      Slist = mk.ndarray(mk[
        mk[1.0,  0.0,  1.0,  0.0  ,  1.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.425]
      ]).transpose(),
      dt = 0.1,
      intRes = 8
    )
    assertEquals(
      mk.d2arrayFromFile("ForwardDynamicsTrajectory-Theta.csv"),
      thetamat,
      "$thetamat not equal ${mk.d2arrayFromFile("ForwardDynamicsTrajectory-Theta.csv")}"
    )
    assertEquals(
      mk.d2arrayFromFile("ForwardDynamicsTrajectory-DTheta.csv"),
      dthetamat
    )

    // Output using kandy to plot the joint angle/velocities
    val N = 10
    val Tf = 10 * 0.1
    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
    plotNVector("ForwardDynamicsTrajectory", timestamps, thetamat.toListD2() to "Theta", dthetamat.toListD2() to "DTheta")
  }

  @Test
  fun `test SimulateControl`() {
    val thetalist = mk.ndarray(mk[0.1, 0.1, 0.1])
    val dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3])
    // Initialize robot description (Example with 3 links)
    val g = mk.ndarray(mk[0.0, 0.0, -9.8])
    // Create a trajectory to follow
    val thetaend = mk.ndarray(mk[PI/2, PI, 1.5*PI])
    val Tf = 1.0
    val N = (Tf / 0.01).roundToInt()
    val method = 5
    val traj = JointTrajectory(thetalist, thetaend, Tf, N, method)
    val thetamatd = traj
    val dthetamatd = mk.zeros<Double>(N, 3)
    val ddthetamatd = mk.zeros<Double>(N, 3)
    val dt = Tf / (N - 1.0)
    for (i in (0..<traj.shape[0] - 1)) {
      dthetamatd[i + 1] = (thetamatd[i + 1]-thetamatd[i]) / dt
      ddthetamatd[i + 1] = (dthetamatd[i + 1]-dthetamatd[i]) / dt
    }
    // Possibly wrong robot description (Example with 3 links)
    val gtilde = mk.ndarray(mk[0.8, 0.2, -8.8])
    val Gtildelist = mk.diagonal(mk[0.1, 0.1, 0.1, 4.0, 4.0, 4.0]).reshape(1, 6, 6) cat
      mk.diagonal(mk[0.3, 0.3, 0.1, 9.0, 9.0, 9.0]).reshape(1, 6, 6) cat
      mk.diagonal(mk[0.1, 0.1, 0.1, 3.0, 3.0, 3.0]).reshape(1, 6, 6)
    val Mtildelist = mk.ndarray(mk[
      mk[
        mk[1.0, 0.0, 0.0, 0.0],
        mk[0.0, 1.0, 0.0, 0.0],
        mk[0.0, 0.0, 1.0, 0.1],
        mk[0.0, 0.0, 0.0, 1.0]
      ],
      mk[
        mk[ 0.0, 0.0, 1.0, 0.3],
        mk[ 0.0, 1.0, 0.0, 0.2],
        mk[-1.0, 0.0, 0.0, 0.0],
        mk[ 0.0, 0.0, 0.0, 1.0]
      ],
      mk[
        mk[1.0, 0.0, 0.0,  0.0],
        mk[0.0, 1.0, 0.0, -0.2],
        mk[0.0, 0.0, 1.0,  0.4],
        mk[0.0, 0.0, 0.0,  1.0]
      ],
      mk[
        mk[1.0, 0.0, 0.0, 0.0],
        mk[0.0, 1.0, 0.0, 0.0],
        mk[0.0, 0.0, 1.0, 0.2],
        mk[0.0, 0.0, 0.0, 1.0]
      ]
    ])
    val Ftipmat = mk.ones<Double>(traj.shape[0], 6)
    val Kp = 20.0
    val Ki = 10.0
    val Kd = 18.0
    val intRes = 8
    val (taumat, thetamat) = SimulateControl(
      thetalist,
      dthetalist,
      g,
      Ftipmat,
      Mlist = mk.ndarray(mk[
        mk[
          mk[1.0, 0.0, 0.0, 0.0     ],
          mk[0.0, 1.0, 0.0, 0.0     ],
          mk[0.0, 0.0, 1.0, 0.089159],
          mk[0.0, 0.0, 0.0, 1.0     ]
        ],
        mk[
          mk[ 0.0, 0.0, 1.0, 0.28   ],
          mk[ 0.0, 1.0, 0.0, 0.13585],
          mk[-1.0, 0.0, 0.0, 0.0    ],
          mk[ 0.0, 0.0, 0.0, 1.0    ]
        ],
        mk[
          mk[1.0, 0.0, 0.0,  0.0   ],
          mk[0.0, 1.0, 0.0, -0.1197],
          mk[0.0, 0.0, 1.0,  0.395 ],
          mk[0.0, 0.0, 0.0,  1.0   ]
        ],
        mk[
          mk[1.0, 0.0, 0.0, 0.0    ],
          mk[0.0, 1.0, 0.0, 0.0    ],
          mk[0.0, 0.0, 1.0, 0.14225],
          mk[0.0, 0.0, 0.0, 1.0    ]
        ]
      ]),
      Glist = mk.diagonal(mk[0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]).reshape(1, 6, 6) cat
        mk.diagonal(mk[0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275]).reshape(1, 6, 6),
      Slist = mk.ndarray(mk[
        mk[1.0,  0.0,  1.0,  0.0  ,  1.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.0  ],
        mk[0.0,  1.0,  0.0, -0.089,  0.0,  0.425]
      ]).transpose(),
      thetamatd,
      dthetamatd,
      ddthetamatd,
      gtilde,
      Mtildelist,
      Gtildelist,
      Kp,
      Ki,
      Kd,
      dt,
      intRes
    )
    assertEquals(
      mk.d2arrayFromFile("SimulateControl-ThetaActual.csv"),
      thetamat
    )
    assertEquals(
      mk.d2arrayFromFile("SimulateControl-ThetaDesired.csv"),
      thetamatd
    )

    // Output using kandy to plot
    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
    plotNVector("SimulateControl", timestamps, thetamat.toListD2() to "Actual Theta", thetamatd.toListD2() to "Desired Theta")
  }
}

fun plotNVector(name: String, timestamps: List<Double>, vararg histories: Pair<List<List<Double>>, String>) {
  val n = histories.sumOf{ it.first.first().size }
  val map = mapOf(
    "t" to (1..n).flatMap { timestamps },
    "values" to histories.flatMap { history -> (0..<history.first.first().size).flatMap { i -> history.first.map { it[i] } }},
    "labels" to histories.flatMap { history -> (1..history.first.first().size).flatMap { i -> timestamps.map { "${history.second}$i" } }}
  )
  map.plot {
    line {
      x("t")
      y("values")
      color("labels")
    }
  }.save("${name}Plot.png")
}

fun Multik.d2arrayFromFile(filename: String): D2Array<Double> {
  return mk.ndarray(checkNotNull(this::class.java.classLoader.getResource(filename)).readText().split("\n").map { it.split(",").map { it.toDouble() } })
}

fun MultiArray<Double, D2>.toFile(filename: String) {
  val csv = toListD2().joinToString("\n") { it.joinToString(",") }
  File(filename).writeText(csv)
}
