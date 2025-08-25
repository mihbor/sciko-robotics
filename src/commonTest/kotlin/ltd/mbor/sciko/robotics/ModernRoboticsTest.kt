package ltd.mbor.sciko.robotics

import com.ionspin.kotlin.bignum.decimal.DecimalMode
import com.ionspin.kotlin.bignum.decimal.RoundingMode
import com.ionspin.kotlin.bignum.decimal.toBigDecimal
import ltd.mbor.sciko.linalg.diagonal
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

fun Double.round(scale: Long = 8) = toBigDecimal().roundToDigitPositionAfterDecimalPoint(digitPosition = scale, roundingMode = RoundingMode.ROUND_HALF_CEILING).doubleValue(false)
fun Double.roundSignificand(decimalPrecision: Long) = toBigDecimal().roundSignificand(DecimalMode(decimalPrecision = decimalPrecision, roundingMode = RoundingMode.ROUND_HALF_CEILING)).doubleValue(false)

fun <D: Dimension> MultiArray<Double, D>.round(scale: Long = 8) = map { it.round(scale) }
fun <D: Dimension> MultiArray<Double, D>.roundSignificand(decimalPrecision: Long) = map { it.roundSignificand(decimalPrecision) }

class ModernRoboticsTest {
  // *** BASIC HELPER FUNCTIONS ***

  @Test
  fun testNearZero() {
    assertEquals(true, NearZero(1e-7))
  }

  @Test
  fun testNormalize() {
    assertEquals(mk.ndarray(mk[0.26726124, 0.53452248, 0.80178373]), Normalize(mk.ndarray(mk[1.0, 2.0, 3.0])).round())
  }

  // *** CHAPTER 3: RIGID-BODY MOTIONS ***

  @Test
  fun testRotInv() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0.0,  1.0,  0.0],
        mk[ 0.0,  0.0,  1.0],
        mk[ 1.0,  0.0,  0.0]
      ]),
      RotInv(mk.ndarray(mk[
        mk[ 0.0,  0.0,  1.0],
        mk[ 1.0,  0.0,  0.0],
        mk[ 0.0,  1.0,  0.0]
      ]))
    )
  }

  @Test
  fun testVecToso3() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0.0, -3.0,  2.0],
        mk[ 3.0,  0.0, -1.0],
        mk[-2.0,  1.0,  0.0]
      ]),
      VecToso3(mk.ndarray(mk[1.0, 2.0, 3.0]))
    )
  }

  @Test
  fun testso3ToVec() {
    assertEquals(
      mk.ndarray(mk[1.0, 2.0, 3.0]),
      so3ToVec(
        mk.ndarray(mk[
          mk[ 0.0, -3.0,  2.0],
          mk[ 3.0,  0.0, -1.0],
          mk[-2.0,  1.0,  0.0]
        ])
      ),
    )
  }

  @Test
  fun testAxisAng3() {
    assertEquals(
      mk.ndarray(mk[0.26726124, 0.53452248, 0.80178373]) to 3.7416573867739413,
      AxisAng3(mk.ndarray(mk[1.0, 2.0, 3.0])).let { it.first.round() to it.second }
    )
  }

  @Test
  fun testMatrixExp3() {
    assertEquals(
      mk.ndarray(mk[
        mk[-0.69492056,  0.71352099,  0.08929286],
        mk[-0.19200697, -0.30378504,  0.93319235],
        mk[ 0.69297817,  0.6313497 ,  0.34810748]
      ]),
      MatrixExp3(
        mk.ndarray(mk[
          mk[ 0, -3,  2],
          mk[ 3,  0, -1],
          mk[-2,  1,  0]
        ]).map { it.toDouble() }
      ).round()
    )
  }

  @Test
  fun testMatrixLog3() {
    assertEquals(
      mk.ndarray(mk[
        mk[        0.0, -1.20919958,  1.20919958],
        mk[ 1.20919958,         0.0, -1.20919958],
        mk[-1.20919958,  1.20919958,         0.0]
      ]),
      MatrixLog3(
        mk.ndarray(mk[
          mk[0, 0, 1],
          mk[1, 0, 0],
          mk[0, 1, 0]
        ]).map { it.toDouble() }
      ).round()
    )
  }

  @Test
  fun testRpToTrans() {
    assertEquals(
      mk.ndarray(mk[
        mk[1,  0,  0,  1],
        mk[0,  0, -1,  2],
        mk[0,  1,  0,  5],
        mk[0,  0,  0,  1]
      ]).map { it.toDouble() },
      RpToTrans(
        R = mk.ndarray(mk[
          mk[1,  0,  0],
          mk[0,  0, -1],
          mk[0,  1,  0]
        ]).map { it.toDouble() },
        p = mk.ndarray(mk[1, 2, 5]).map { it.toDouble() }
      )
    )
  }

  @Test
  fun testTransToRp() {
    assertEquals(
      mk.ndarray(mk[
        mk[1,  0,  0],
        mk[0,  0, -1],
        mk[0,  1,  0]
      ]).map { it.toDouble() }
      to mk.ndarray(mk[0, 0, 3]).map { it.toDouble() },
      TransToRp(
        mk.ndarray(mk[
          mk[1,  0,  0,  0],
          mk[0,  0, -1,  0],
          mk[0,  1,  0,  3],
          mk[0,  0,  0,  1]
        ]).map { it.toDouble() }
      )
    )
  }

  @Test
  fun testTransInv() {
    assertEquals(
      mk.ndarray(mk[
        mk[1,  0,  0,  0],
        mk[0,  0,  1, -3],
        mk[0, -1,  0,  0],
        mk[0,  0,  0,  1]
      ]).map { it.toDouble() },
      TransInv(mk.ndarray(mk[
        mk[1,  0,  0,  0],
        mk[0,  0, -1,  0],
        mk[0,  1,  0,  3],
        mk[0,  0,  0,  1]
      ]).map { it.toDouble() })
    )
  }

  @Test
  fun testVecTose3() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0, -3,  2,  4],
        mk[ 3,  0, -1,  5],
        mk[-2,  1,  0,  6],
        mk[ 0,  0,  0,  0]
      ]).map { it.toDouble() },
      VecTose3(
        mk.ndarray(mk[1, 2, 3, 4, 5, 6]).map { it.toDouble() }
      )
    )
  }

  @Test
  fun testse3ToVec() {
    assertEquals(
      mk.ndarray(mk[1, 2, 3, 4, 5, 6]).map { it.toDouble() },
      se3ToVec(
        mk.ndarray(mk[
          mk[ 0, -3,  2,  4],
          mk[ 3,  0, -1,  5],
          mk[-2,  1,  0,  6],
          mk[ 0,  0,  0,  0]
        ]).map { it.toDouble() }
      )
    )
  }

  @Test
  fun testAdjoint() {
    assertEquals(
      mk.ndarray(mk[
        mk[1,  0,  0,  0,  0,  0],
        mk[0,  0, -1,  0,  0,  0],
        mk[0,  1,  0,  0,  0,  0],
        mk[0,  0,  3,  1,  0,  0],
        mk[3,  0,  0,  0,  0, -1],
        mk[0,  0,  0,  0,  1,  0]
      ]).map { it.toDouble() },
      Adjoint(
        mk.ndarray(mk[
          mk[1,  0,  0,  0],
          mk[0,  0, -1,  0],
          mk[0,  1,  0,  3],
          mk[0,  0,  0,  1]
        ]).map { it.toDouble() }
      )
    )
  }

  @Test
  fun testScrewToAxis() {
    assertEquals(
      mk.ndarray(mk[0, 0, 1, 0, -3, 2]).map { it.toDouble() },
      ScrewToAxis(
        q = mk.ndarray(mk[3, 0, 0]).map { it.toDouble() },
        s = mk.ndarray(mk[0, 0, 1]).map { it.toDouble() },
        h = 2.0
      )
    )
  }

  @Test
  fun testAxisAng6() {
    assertEquals(
      mk.ndarray(mk[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]) to 1.0,
      AxisAng6(mk.ndarray(mk[1, 0, 0, 1, 2, 3]).map { it.toDouble() })
    )
  }

  @Test
  fun testMatrixExp6() {
    assertEquals(
      mk.ndarray(mk[
        mk[1.0,  0.0,  0.0,  0.0],
        mk[0.0,  0.0, -1.0,  0.0],
        mk[0.0,  1.0,  0.0,  3.0],
        mk[0.0,  0.0,  0.0,  1.0]
      ]),
      MatrixExp6(mk.ndarray(mk[
        mk[0.0,  0.0,   0.0,    0.0],
        mk[0.0,  0.0, -PI/2, 3*PI/4],
        mk[0.0, PI/2,   0.0, 3*PI/4],
        mk[0.0,  0.0,   0.0,    0.0]
      ])).round()
    )
  }

  @Test
  fun testMatrixLog6() {
    assertEquals(
      mk.ndarray(mk[
        mk[0.0,  0.0,   0.0,    0.0],
        mk[0.0,  0.0, -PI/2, 3*PI/4],
        mk[0.0, PI/2,   0.0, 3*PI/4],
        mk[0.0,  0.0,   0.0,    0.0]
      ]).round(),
      MatrixLog6(mk.ndarray(mk[
        mk[1,  0,  0,  0],
        mk[0,  0, -1,  0],
        mk[0,  1,  0,  3],
        mk[0,  0,  0,  1]
      ]).map { it.toDouble() }).round()
    )
  }

  @Test
  fun testProjectToSO3() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0.67901136,  0.14894516,  0.71885945],
        mk[ 0.37320708,  0.77319584, -0.51272279],
        mk[-0.63218672,  0.61642804,  0.46942137]
      ]),
      ProjectToSO3(mk.ndarray(mk[
        mk[ 0.675,  0.150,  0.720],
        mk[ 0.370,  0.771, -0.511],
        mk[-0.630,  0.619,  0.472]
      ])).round()
    )
  }

  @Test
  fun testProjectToSE3() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0.67901136,  0.14894516,  0.71885945,  1.2],
        mk[ 0.37320708,  0.77319584, -0.51272279,  5.4],
        mk[-0.63218672,  0.61642804,  0.46942137,  3.6],
        mk[ 0.0       ,  0.0       ,  0.0       ,  1.0]
      ]),
      ProjectToSE3(mk.ndarray(mk[
        mk[ 0.675,  0.150,  0.720,  1.2],
        mk[ 0.370,  0.771, -0.511,  5.4],
        mk[-0.630,  0.619,  0.472,  3.6],
        mk[ 0.003,  0.002,  0.010,  0.9]
      ])).round()
    )
  }

  @Test
  fun testDistanceToSO3() {
    assertEquals(
      0.08835,
      DistanceToSO3(
        mk.ndarray(mk[
          mk[ 1.0,  0.0,  0.0 ],
          mk[ 0.0,  0.1, -0.95],
          mk[ 0.0,  1.0,  0.1 ]
        ])
      ).round(5)
    )
  }

  @Test
  fun testDistanceToSE3() {
    assertEquals(
      0.134931,
      DistanceToSE3(
        mk.ndarray(mk[
          mk[ 1.0,  0.0,  0.0 ,  1.2 ],
          mk[ 0.0,  0.1, -0.95,  1.5 ],
          mk[ 0.0,  1.0,  0.1 , -0.9 ],
          mk[ 0.0,  0.0,  0.1 ,  0.98]
        ])
      ).round(6)
    )
  }

  @Test
  fun testTestIfSO3() {
    assertFalse(
      TestIfSO3(
        mk.ndarray(
          mk[
            mk[1.0,  0.0,  0.0 ],
            mk[0.0,  0.1, -0.95],
            mk[0.0,  1.0,  0.1 ]
          ]
        )
      )
    )
  }

  @Test
  fun testTestIfSE3() {
    assertFalse(
      TestIfSE3(
        mk.ndarray(mk[
          mk[1.0,  0.0,  0.0 ,  1.2 ],
          mk[0.0,  0.1, -0.95,  1.5 ],
          mk[0.0,  1.0,  0.1 , -0.9 ],
          mk[0.0,  0.0,  0.1 ,  0.98]
        ])
      )
    )
  }

  // *** CHAPTER 4: FORWARD KINEMATICS ***

  @Test
  fun testFKinBody() {
    assertEquals(
      mk.ndarray(mk[
        mk[0.0,  1.0,  0.0, -5.0       ],
        mk[1.0,  0.0,  0.0,  4.0       ],
        mk[0.0,  0.0, -1.0,  1.68584073],
        mk[0.0,  0.0,  0.0,  1.0       ]
      ]),
      FKinBody(
        M = mk.ndarray(mk[
          mk[-1,  0,  0,  0],
          mk[ 0,  1,  0,  6],
          mk[ 0,  0, -1,  2],
          mk[ 0,  0,  0,  1]
        ]).map { it.toDouble() },
        Blist = mk.ndarray(mk[
          mk[0.0,  0.0, -1.0,  2.0,  0.0,  0.0],
          mk[0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
          mk[0.0,  0.0,  1.0,  0.0,  0.0,  0.1]
        ]).transpose(),
        thetalist = mk.ndarray(mk[PI / 2.0, 3.0, PI])
      ).round()
    )
  }

  @Test
  fun testFKinSpace() {
    assertEquals(
      mk.ndarray(mk[
        mk[0.0,  1.0,  0.0, -5.0       ],
        mk[1.0,  0.0,  0.0,  4.0       ],
        mk[0.0,  0.0, -1.0,  1.68584073],
        mk[0.0,  0.0,  0.0,  1.0       ]
      ]),
      FKinSpace(
        M = mk.ndarray(mk[
          mk[-1,  0,  0,  0],
          mk[ 0,  1,  0,  6],
          mk[ 0,  0, -1,  2],
          mk[ 0,  0,  0,  1]
        ]).map { it.toDouble() },
        Slist = mk.ndarray(mk[
          mk[0.0,  0.0,  1.0,  4.0,  0.0,  0.0],
          mk[0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
          mk[0.0,  0.0, -1.0, -6.0,  0.0, -0.1]
        ]).transpose(),
        thetalist = mk.ndarray(mk[PI / 2.0, 3.0, PI])
      ).round()
    )
  }

  // *** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***

  @Test
  fun testJacobianBody() {
    assertEquals(
      mk.ndarray(mk[
        mk[-0.04528405,  0.99500417,  0.0       ,  1.0],
        mk[ 0.74359313,  0.09304865,  0.36235775,  0.0],
        mk[-0.66709716,  0.03617541, -0.93203909,  0.0],
        mk[ 2.32586047,  1.66809   ,  0.56410831,  0.2],
        mk[-1.44321167,  2.94561275,  1.43306521,  0.3],
        mk[-2.06639565,  1.82881722, -1.58868628,  0.4]
      ]),
      JacobianBody(
        Blist = mk.ndarray(mk[
          mk[0.0,  0.0,  1.0,  0.0,  0.2,  0.2],
          mk[1.0,  0.0,  0.0,  2.0,  0.0,  3.0],
          mk[0.0,  1.0,  0.0,  0.0,  2.0,  1.0],
          mk[1.0,  0.0,  0.0,  0.2,  0.3,  0.4]
        ]).transpose(),
        thetalist = mk.ndarray(mk[0.2, 1.1, 0.1, 1.2])
      ).round()
    )
  }

  @Test
  fun testJacobianSpace() {
    assertEquals(
      mk.ndarray(mk[
        mk[0.0,  0.98006658, -0.09011564,  0.95749426],
        mk[0.0,  0.19866933,  0.4445544 ,  0.28487557],
        mk[1.0,  0.0       ,  0.89120736, -0.04528405],
        mk[0.0,  1.95218638, -2.21635216, -0.51161537],
        mk[0.2,  0.43654132, -2.43712573,  2.77535713],
        mk[0.2,  2.96026613,  3.23573065,  2.22512443]
      ]),
      JacobianSpace(
        Slist = mk.ndarray(mk[
          mk[0.0,  0.0,  1.0,  0.0,  0.2,  0.2],
          mk[1.0,  0.0,  0.0,  2.0,  0.0,  3.0],
          mk[0.0,  1.0,  0.0,  0.0,  2.0,  1.0],
          mk[1.0,  0.0,  0.0,  0.2,  0.3,  0.4]
        ]).transpose(),
        thetalist = mk.ndarray(mk[0.2, 1.1, 0.1, 1.2])
      ).round()
    )
  }

  @Test
  fun testIKinBody() {
    val (result, converged) = IKinBody(
      Blist = mk.ndarray(mk[
        mk[0.0,  0.0, -1.0,  2.0,  0.0,  0.0],
        mk[0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
        mk[0.0,  0.0,  1.0,  0.0,  0.0,  0.1]
      ]).transpose(),
      M = mk.ndarray(mk[
        mk[-1,  0,  0,  0],
        mk[ 0,  1,  0,  6],
        mk[ 0,  0, -1,  2],
        mk[ 0,  0,  0,  1]
      ]).map { it.toDouble() },
      T = mk.ndarray(mk[
        mk[0.0,  1.0,  0.0, -5.0   ],
        mk[1.0,  0.0,  0.0,  4.0   ],
        mk[0.0,  0.0, -1.0,  1.6858],
        mk[0.0,  0.0,  0.0,  1.0   ]
      ]),
      thetalist0 = mk.ndarray(mk[1.5, 2.5, 3.0]),
      eomg = 0.01,
      ev = 0.001
    )
    assertTrue(converged)
    assertEquals(
      mk.ndarray(mk[1.57073819, 2.999667, 3.14153913]),
      result.round()
    )
  }

  @Test
  fun testIKinSpace() {
    val (result, converged) = IKinSpace(
      Slist = mk.ndarray(mk[
        mk[0.0,  0.0,  1.0,  4.0,  0.0,  0.0],
        mk[0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
        mk[0.0,  0.0, -1.0, -6.0,  0.0, -0.1]
      ]).transpose(),
      M = mk.ndarray(mk[
        mk[-1,  0,  0,  0],
        mk[ 0,  1,  0,  6],
        mk[ 0,  0, -1,  2],
        mk[ 0,  0,  0,  1]
      ]).map { it.toDouble() },
      T = mk.ndarray(mk[
        mk[0.0,  1.0,  0.0, -5.0   ],
        mk[1.0,  0.0,  0.0,  4.0   ],
        mk[0.0,  0.0, -1.0,  1.6858],
        mk[0.0,  0.0,  0.0,  1.0   ]
      ]),
      thetalist0 = mk.ndarray(mk[1.5, 2.5, 3.0]),
      eomg = 0.01,
      ev = 0.001
    )
    assertTrue(converged)
    assertEquals(
      mk.ndarray(mk[1.57073783, 2.99966384, 3.1415342]),
      result.round()
    )
  }

  // *** CHAPTER 8: DYNAMICS OF OPEN CHAINS ***

  @Test
  fun testad() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 0, -3,  2,  0,  0,  0],
        mk[ 3,  0, -1,  0,  0,  0],
        mk[-2,  1,  0,  0,  0,  0],
        mk[ 0, -6,  5,  0, -3,  2],
        mk[ 6,  0, -4,  3,  0, -1],
        mk[-5,  4,  0, -2,  1,  0]
      ]).map { it.toDouble() },
      ad(mk.ndarray(mk[1, 2, 3, 4, 5, 6]).map { it.toDouble() })
    )
  }

  @Test
  fun testInverseDynamics() {
    assertEquals(
      mk.ndarray(mk[74.69616155, -33.06766016, -3.23057314]),
      InverseDynamics(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
        ddthetalist = mk.ndarray(mk[2.0, 1.5, 1.0]),
        g = mk.ndarray(mk[0.0, 0.0, -9.8]),
        Ftip = mk.ndarray(mk[1, 1, 1, 1, 1, 1]).map { it.toDouble() },
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
        ]).transpose()
      ).round()
    )
  }

  @Test
  fun testMassMatrix() {
    assertEquals(
      mk.ndarray(mk[
        mk[ 2.25433380e+01, -3.07146754e-01, -7.18426391e-03],
        mk[-3.07146754e-01,  1.96850717e+00,  4.32157368e-01],
        mk[-7.18426391e-03,  4.32157368e-01,  1.91630858e-01]
      ]),
      MassMatrix(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
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
        ]).transpose()
      ).roundSignificand(9)
    )
  }

  @Test
  fun testVelQuadraticForces() {
    assertEquals(
      mk.ndarray(mk[0.26453118, -0.05505157, -0.00689132]),
      VelQuadraticForces(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
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
        ]).transpose()
      ).round()
    )
  }

  @Test
  fun testGravityForces() {
    assertEquals(
      mk.ndarray(mk[28.40331262, -37.64094817, -5.4415892]),
      GravityForces(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        g = mk.ndarray(mk[0.0, 0.0, -9.8]),
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
        ]).transpose()
      ).round()
    )
  }

  @Test
  fun testEndEffectorForces() {
    assertEquals(
      mk.ndarray(mk[1.40954608, 1.85771497, 1.392409]),
      EndEffectorForces(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        Ftip = mk.ndarray(mk[1, 1, 1, 1, 1, 1]).map { it.toDouble() },
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
        ]).transpose()
      ).round()
    )
  }

  @Test
  fun testForwardDynamics() {
    assertEquals(
      mk.ndarray(mk[-0.97392907, 25.58466784, -32.91499212]),
      ForwardDynamics(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
        taulist = mk.ndarray(mk[0.5, 0.6, 0.7]),
        g = mk.ndarray(mk[0.0, 0.0, -9.8]),
        Ftip = mk.ndarray(mk[1, 1, 1, 1, 1, 1]).map { it.toDouble() },
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
        ]).transpose()
      ).round()
    )
  }

  @Test
  fun testEulerStep() {
    assertEquals(
      mk.ndarray(mk[0.11, 0.12, 0.13]) to mk.ndarray(mk[0.3, 0.35, 0.4]),
      EulerStep(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
        ddthetalist = mk.ndarray(mk[2.0, 1.5, 1.0]),
        dt = 0.1
      ).let { it.first.round() to it.second.round() }
    )
  }

  @Test
  fun testInverseDynamicsTrajectory() {

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
// Only implemented for jvm target
//    assertEquals(
//      mk.d2arrayFromFile("InverseDynamicsTrajectory.csv"),
//      taumat
//    )
//
//    //Output using kandy to plot the joint forces/torques
//    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
//    plotNVector("InverseDynamicsTrajectory", timestamps, taumat.toListD2() to "Tau")
  }

  @Test
  fun testForwardDynamicsTrajectory() {
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
// Only implemented for jvm target
//    assertEquals(
//      mk.d2arrayFromFile("ForwardDynamicsTrajectory-Theta.csv"),
//      thetamat
//    )
//    assertEquals(
//      mk.d2arrayFromFile("ForwardDynamicsTrajectory-DTheta.csv"),
//      dthetamat
//    )
//
//    // Output using kandy to plot the joint angle/velocities
//    val N = 10
//    val Tf = 10 * 0.1
//    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
//    plotNVector("ForwardDynamicsTrajectory", timestamps, thetamat.toListD2() to "Theta", dthetamat.toListD2() to "DTheta")
  }

  // *** CHAPTER 9: TRAJECTORY GENERATION ***

  @Test
  fun testCubicTimeScaling() {
    assertEquals(
      0.216,
      CubicTimeScaling(Tf = 2.0, t = 0.6).round()
    )
  }

  @Test
  fun testQuinticTimeScaling() {
    assertEquals(
      0.16308,
      QuinticTimeScaling(Tf = 2.0, t = 0.6).round()
    )
  }

  @Test
  fun testJointTrajectory() {
    assertEquals(
      mk.ndarray(mk[
        mk[1.0   , 0.0  , 0.0   , 1.0   , 1.0  , 0.2   , 0.0   , 1.0],
        mk[1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1.0],
        mk[1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1.0],
        mk[1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1.0],
        mk[1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1.0],
        mk[1.2   , 0.5  , 0.6   , 1.1   , 2.0  , 2.0   , 0.9   , 1.0],
      ]),
      JointTrajectory(
        thetastart = mk.ndarray(mk[1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.0, 1.0]),
        thetaend = mk.ndarray(mk[1.2, 0.5, 0.6, 1.1, 2.0, 2.0, 0.9, 1.0]),
        Tf = 4.0,
        N = 6,
        method = 3
      ).round()
    )
  }

  @Test
  fun testScrewTrajectory() {
    assertEquals(
      listOf(
        mk.ndarray(mk[
          mk[1, 0, 0, 1],
          mk[0, 1, 0, 0],
          mk[0, 0, 1, 1],
          mk[0, 0, 0, 1],
        ]).map{ it.toDouble() },
        mk.ndarray(mk[
          mk[ 0.904,-0.25 , 0.346, 0.441],
          mk[ 0.346, 0.904,-0.25 , 0.529],
          mk[-0.25 , 0.346, 0.904, 1.601],
          mk[ 0.0  , 0.0  , 0.0  , 1.0  ],
        ]),
        mk.ndarray(mk[
          mk[ 0.346,-0.25 , 0.904,-0.117],
          mk[ 0.904, 0.346,-0.25 , 0.473],
          mk[-0.25 , 0.904, 0.346, 3.274],
          mk[ 0.0  , 0.0  , 0.0  , 1.0  ],
        ]),
        mk.ndarray(mk[
          mk[0.0, 0.0, 1.0, 0.1],
          mk[1.0, 0.0, 0.0, 0.0],
          mk[0.0, 1.0, 0.0, 4.1],
          mk[0.0, 0.0, 0.0, 1.0],
        ])
      ),
      ScrewTrajectory(
        Xstart = mk.ndarray(mk[
          mk[1, 0, 0, 1],
          mk[0, 1, 0, 0],
          mk[0, 0, 1, 1],
          mk[0, 0, 0, 1]
        ]).map{ it.toDouble() },
        Xend = mk.ndarray(mk[
          mk[0.0, 0.0, 1.0, 0.1],
          mk[1.0, 0.0, 0.0, 0.0],
          mk[0.0, 1.0, 0.0, 4.1],
          mk[0.0, 0.0, 0.0, 1.0]
        ]),
        Tf = 5.0,
        N = 4,
        method = 3
      ).map{ it.round(scale=3) }
    )
  }

  @Test
  fun testCartesianTrajectory() {
    assertEquals(
      listOf(
        mk.ndarray(mk[
          mk[1, 0, 0, 1],
          mk[0, 1, 0, 0],
          mk[0, 0, 1, 1],
          mk[0, 0, 0, 1],
        ]).map{ it.toDouble() },
        mk.ndarray(mk[
          mk[ 0.937, -0.214,  0.277, 0.811],
          mk[ 0.277,  0.937, -0.214, 0.0  ],
          mk[-0.214,  0.277,  0.937, 1.651],
          mk[ 0.0  ,  0.0  ,  0.0  , 1.0  ],
        ]),
        mk.ndarray(mk[
          mk[ 0.277, -0.214,  0.937, 0.289],
          mk[ 0.937,  0.277, -0.214, 0.0  ],
          mk[-0.214,  0.937,  0.277, 3.449],
          mk[ 0.0  ,  0.0  ,  0.0  , 1.0  ],
        ]),
        mk.ndarray(mk[
          mk[0.0, 0.0, 1.0, 0.1],
          mk[1.0, 0.0, 0.0, 0.0],
          mk[0.0, 1.0, 0.0, 4.1],
          mk[0.0, 0.0, 0.0, 1.0],
        ])
      ),
      CartesianTrajectory(
        Xstart = mk.ndarray(mk[
          mk[1, 0, 0, 1],
          mk[0, 1, 0, 0],
          mk[0, 0, 1, 1],
          mk[0, 0, 0, 1]
        ]).map{ it.toDouble() },
        Xend = mk.ndarray(mk[
          mk[0.0, 0.0, 1.0, 0.1],
          mk[1.0, 0.0, 0.0, 0.0],
          mk[0.0, 1.0, 0.0, 4.1],
          mk[0.0, 0.0, 0.0, 1.0]
        ]),
        Tf = 5.0,
        N = 4,
        method = 5
      ).map{ it.round(3) }
    )
  }

  @Test
  fun testComputedTorque() {
    assertEquals(
      mk.ndarray(mk[133.00525246, -29.94223324, -3.03276856]),
      ComputedTorque(
        thetalist = mk.ndarray(mk[0.1, 0.1, 0.1]),
        dthetalist = mk.ndarray(mk[0.1, 0.2, 0.3]),
        eint = mk.ndarray(mk[0.2, 0.2, 0.2]),
        g = mk.ndarray(mk[0.0, 0.0, -9.8]),
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
        thetalistd = mk.ndarray(mk[1.0, 1.0, 1.0]),
        dthetalistd = mk.ndarray(mk[2.0, 1.2, 2.0]),
        ddthetalistd = mk.ndarray(mk[0.1, 0.1, 0.1]),
        Kp = 1.3,
        Ki = 1.2,
        Kd = 1.1
      ).round()
    )
  }

  @Test
  fun testSimulateControl() {
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
// Only implemented for jvm target
//    assertEquals(
//      mk.d2arrayFromFile("SimulateControl-ThetaActual.csv"),
//      thetamat
//    )
//    assertEquals(
//      mk.d2arrayFromFile("SimulateControl-ThetaDesired.csv"),
//      thetamatd
//    )
//
//    // Output using kandy to plot
//    val timestamps = mk.linspace<Double>(0.0, Tf, N).toList()
//    plotNVector("SimulateControl", timestamps, thetamat.toListD2() to "Actual Theta", thetamatd.toListD2() to "Desired Theta")
  }
}
