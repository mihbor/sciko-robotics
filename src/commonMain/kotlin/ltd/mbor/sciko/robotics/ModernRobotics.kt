package ltd.mbor.sciko.robotics

import ltd.mbor.sciko.linalg.det
import ltd.mbor.sciko.linalg.pinv
import ltd.mbor.sciko.linalg.svd
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

// *** BASIC HELPER FUNCTIONS ***

fun MultiArray<Double, D1>.norm() = mk.linalg.norm(reshape(shape[0],1))

fun MultiArray<Double, D2>.trace(): Double {
  if (shape[0] != shape[1]) throw IllegalStateException("matrix not square")
  return (0..<shape[0]).map{this[it, it]}.sum()
}

fun NearZero(z: Double) = abs(z) < 1e-6

fun Normalize(V: MultiArray<Double, D1>): MultiArray<Double, D1> {
  return V / V.norm()
}

// *** CHAPTER 3: RIGID-BODY MOTIONS ***

fun RotInv(R: MultiArray<Double, D2>) = R.transpose()

fun VecToso3(omg: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return mk.ndarray(mk[
    mk[0.0    , -omg[2],  omg[1]],
    mk[omg[2] ,     0.0, -omg[0]],
    mk[-omg[1],  omg[0],    0.0]])
}

fun so3ToVec(so3mat: MultiArray<Double, D2>): MultiArray<Double, D1> {
  return mk.ndarray(mk[so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])
}

fun AxisAng3(expc3: MultiArray<Double, D1>): Pair<MultiArray<Double, D1>, Double> {
  return Normalize(expc3) to expc3.norm()
}

fun MatrixExp3(so3mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val omgtheta = so3ToVec(so3mat)
  return if (NearZero(omgtheta.norm())) mk.identity<Double>(3)
  else {
    val (_, theta) = AxisAng3(omgtheta)
    val omgmat = so3mat / theta
    mk.identity<Double>(3) + sin(theta) * omgmat + (1 - cos(theta)) * (omgmat dot omgmat)
  }
}

fun MatrixLog3(R: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val acosinput = (R.trace() - 1) / 2.0
  return if (acosinput >= 1) {
    mk.zeros<Double>(3, 3)
  } else if (acosinput <= -1) {
    val omg =
      if (!NearZero(1 + R[2][2]))
        (1.0 / sqrt(2 * (1 + R[2][2]))) * mk.ndarray(mk[R[0][2], R[1][2], 1 + R[2][2]])
      else if(!NearZero(1 + R[1][1]))
        (1.0 / sqrt(2 * (1 + R[1][1]))) * mk.ndarray(mk[R[0][1], 1 + R[1][1], R[2][1]])
      else
        (1.0 / sqrt(2 * (1 + R[0][0]))) * mk.ndarray(mk[1 + R[0][0], R[1][0], R[2][0]])
    VecToso3(PI * omg)
  } else {
    val theta = acos(acosinput)
    theta / 2.0 / sin(theta) * (R - R.transpose())
  }
}

fun RpToTrans(R: MultiArray<Double, D2>, p: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return R.cat(p.reshape(3, 1), axis=1).cat(mk.ndarray(mk[mk[0.0, 0.0, 0.0, 1.0]]), axis=0)
}

fun TransToRp(T: MultiArray<Double, D2>): Pair<MultiArray<Double, D2>, MultiArray<Double, D1>> {
  return T[0..2, 0..2] to T[0..2, 3]
}

fun TransInv(T: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (R, p) = TransToRp(T)
  val Rt = R.transpose()
  return RpToTrans(Rt, -Rt dot p)
}

fun VecTose3(V: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return VecToso3(V[0..2])
    .cat(V[3..5].reshape(3,1), axis=1)
    .cat(mk.zeros(1,4))
}

fun se3ToVec(se3mat: MultiArray<Double, D2>): MultiArray<Double, D1> {
  return mk.ndarray(mk[
    se3mat[2, 1], se3mat[0, 2], se3mat[1, 0],
    se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]
  ])
}

fun Adjoint(T: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (R, p) = TransToRp(T)
  return R.cat(mk.zeros(3, 3), axis=1).cat((VecToso3(p) dot R).cat(R, axis=1), axis=0)
}

fun ScrewToAxis(q: MultiArray<Double, D1>, s: MultiArray<Double, D1>, h: Double): MultiArray<Double, D1> {
  return s.cat((VecToso3(q) dot s) + h * s, axis=0)
}

fun AxisAng6(expc6: MultiArray<Double, D1>): Pair<MultiArray<Double, D1>, Double> {
  var theta = expc6[0..2].norm()
  if (NearZero(theta))
    theta = expc6[3..5].norm()
  return (expc6 / theta) to theta
}

fun MatrixExp6(se3mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (omega, v) = TransToRp(se3mat)
  val omgtheta = so3ToVec(omega)
  return if (NearZero(omgtheta.norm()))
    mk.identity<Double>(3).cat(v.reshape(3, 1), axis=1)
      .cat(mk.ndarray(mk[mk[0.0, 0.0, 0.0, 1.0]]))
  else {
    val (_, theta) = AxisAng3(omgtheta)
    val omgmat = omega / theta
    MatrixExp3(omega).cat(
      (mk.identity<Double>(3) * theta + (1 - cos(theta)) * omgmat + (theta - sin(theta)) * (omgmat dot omgmat)) dot v.reshape(3, 1) / theta,
      axis=1
    ).cat(mk.ndarray(mk[mk[0.0, 0.0, 0.0, 1.0]]), axis=0)
  }
}

fun MatrixLog6(T: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (R,_) = TransToRp(T)
  val omgmat = MatrixLog3(R)
  return if (omgmat.all{ it == 0.0})
    mk.zeros<Double>(3, 3)
      .cat(mk.ndarray(mk[T[0, 3], T[1, 3], T[2, 3]]).reshape(3,1), axis=1)
      .cat(mk.zeros(1,4), axis=0)
  else {
    val theta = acos((R.trace() - 1.0) / 2.0)
    omgmat.cat(
      (mk.identity<Double>(3) - omgmat / 2.0 + (1.0 / theta - 1.0 / tan(theta / 2.0) / 2.0) * (omgmat dot omgmat) / theta)
        dot mk.ndarray(mk[T[0][3], T[1][3], T[2][3]]).reshape(3,1),
      axis=1
    ).cat(mk.zeros(1,4), axis=0)
  }
}

fun ProjectToSO3(mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (U, s, Vh) = mk.linalg.svd(mat)
  val R = U dot Vh
  if (mk.linalg.det(R) < 0.0) {
    (0..<R.shape[0]).forEach {
      R[it, 2] = -R[it, 2]
    }
  }
  return R
}

fun ProjectToSE3(mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  return RpToTrans(ProjectToSO3(mat[0..<3, 0..<3]), mat[0..<3, 3])
}

fun DistanceToSO3(mat: MultiArray<Double, D2>): Double {
  return if (mk.linalg.det(mat) > 0) mk.linalg.norm((mat.transpose() dot mat) - mk.identity(3))
  else 1e9
}

fun DistanceToSE3(mat: MultiArray<Double, D2>): Double {
  val matR = mat[0..<3, 0..<3]
  return if (mk.linalg.det(matR) > 0)
    mk.linalg.norm(
      (matR.transpose() dot matR).cat(mk.zeros<Double>(3, 1), axis=1)
        .cat(mat[3].reshape(1, 4)) - mk.identity(4)
    )
  else 1e9
}

fun TestIfSO3(mat: MultiArray<Double, D2>): Boolean {
  return abs(DistanceToSO3(mat)) < 1e-3
}

fun TestIfSE3(mat: MultiArray<Double, D2>): Boolean {
  return abs(DistanceToSE3(mat)) < 1e-3
}

// *** CHAPTER 4: FORWARD KINEMATICS ***

fun FKinBody(M: MultiArray<Double, D2>, Blist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  var T = M
  val BlistT = Blist.transpose()
  for (i in 0..<thetalist.size) {
    T = T dot MatrixExp6(VecTose3(BlistT[i].transpose()  * thetalist[i]))
  }
  return T
}

fun FKinSpace(M: MultiArray<Double, D2>, Slist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  var T = M
  val SlistT = Slist.transpose()
  for (i in thetalist.size-1 downTo 0) {
    T = MatrixExp6(VecTose3(SlistT[i].transpose() * thetalist[i])) dot T
  }
  return T
}

// *** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***

fun JacobianBody(Blist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  val BlistT = Blist.transpose()
  val Jb = mk.zeros<Double>(BlistT.shape[0], BlistT.shape[1])
  Jb[thetalist.size-1] = BlistT[thetalist.size-1]
  var T = mk.identity<Double>(4)
  for (i in thetalist.size - 2 downTo 0) {
    T = T dot MatrixExp6(VecTose3(BlistT[i + 1].transpose()  * -thetalist[i + 1]))
    Jb[i] = (Adjoint(T) dot BlistT[i].transpose()).flatten()
  }
  return Jb.transpose()
}

fun JacobianSpace(Slist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  val SlistT = Slist.transpose()
  val Js = mk.zeros<Double>(SlistT.shape[0], SlistT.shape[1])
  Js[0] = SlistT[0]
  var T = mk.identity<Double>(4)
  for (i in 1..<thetalist.size) {
    T = T dot MatrixExp6(VecTose3(SlistT[i - 1].transpose()  * thetalist[i - 1]))
    Js[i] = (Adjoint(T) dot SlistT[i].transpose()).flatten()
  }
  return Js.transpose()
}

fun IKinBody(
  Blist: MultiArray<Double, D2>,
  M: MultiArray<Double, D2>,
  T: MultiArray<Double, D2>,
  thetalist0: MultiArray<Double, D1>,
  eomg: Double,
  ev: Double
): Pair<MultiArray<Double, D1>, Boolean> {

  var thetalist = thetalist0
  var i = 0
  val maxiterations = 20
  var Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) dot T))
  var err = mk.ndarray(mk[Vb[0], Vb[1], Vb[2]]).norm() > eomg || mk.ndarray(mk[Vb[3], Vb[4], Vb[5]]).norm() > ev
  while (err && i < maxiterations) {
    thetalist = thetalist + (mk.linalg.pinv(JacobianBody(Blist, thetalist)) dot Vb)
    i = i + 1
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) dot T))
    err = mk.ndarray(mk[Vb[0], Vb[1], Vb[2]]).norm() > eomg || mk.ndarray(mk[Vb[3], Vb[4], Vb[5]]).norm() > ev
  }
  return thetalist to !err
}

fun IKinSpace(
  Slist: MultiArray<Double, D2>,
  M: MultiArray<Double, D2>,
  T: MultiArray<Double, D2>,
  thetalist0: MultiArray<Double, D1>,
  eomg: Double,
  ev: Double
): Pair<MultiArray<Double, D1>, Boolean> {

  var thetalist = thetalist0
  var i = 0
  val maxiterations = 20
  var Tsb = FKinSpace(M,Slist, thetalist)
  var Vs = Adjoint(Tsb) dot se3ToVec(MatrixLog6(TransInv(Tsb) dot T))
  var err = mk.ndarray(mk[Vs[0], Vs[1], Vs[2]]).norm() > eomg || mk.ndarray(mk[Vs[3], Vs[4], Vs[5]]).norm() > ev
  while (err && i < maxiterations) {
    thetalist = thetalist + (mk.linalg.pinv(JacobianSpace(Slist, thetalist)) dot Vs)
    i = i + 1
    Tsb = FKinSpace(M,Slist, thetalist)
    Vs = Adjoint(Tsb) dot se3ToVec(MatrixLog6(TransInv(Tsb) dot T))
    err = mk.ndarray(mk[Vs[0], Vs[1], Vs[2]]).norm() > eomg || mk.ndarray(mk[Vs[3], Vs[4], Vs[5]]).norm() > ev
  }
  return thetalist to !err
}

// *** CHAPTER 8: DYNAMICS OF OPEN CHAINS ***

fun ad(V: MultiArray<Double, D1>): MultiArray<Double, D2> {
  val omgmat = VecToso3(mk.ndarray(mk[V[0], V[1], V[2]]))
  return omgmat.cat(mk.zeros(3, 3), axis=1).cat(
    VecToso3(mk.ndarray(mk[V[3], V[4], V[5]])).cat(omgmat, axis=1), axis=0)
}

fun InverseDynamics(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  ddthetalist: MultiArray<Double, D1>,
  g: MultiArray<Double, D1>,
  Ftip: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
): MultiArray<Double, D1> {
  val SlistT = Slist.transpose()
  val n = thetalist.size
  var Mi = mk.identity<Double>(4)
  val Ai = mk.zeros<Double>(6, n)
  val AiT = Ai.transpose()
  val AdTi = MutableList<MultiArray<Double, D2>?>(n+1) {null}
  val Vi = mk.zeros<Double>(6, n + 1)
  val ViT = Vi.transpose()
  val Vdi = mk.zeros<Double>(6, n + 1)
  val VdiT = Vdi.transpose()
  VdiT[0] = mk.ndarray(mk[0.0, 0.0, 0.0]).cat(-g).transpose()
  AdTi[n] = Adjoint(TransInv(Mlist[n]))
  var Fi = Ftip
  val taulist = mk.zeros<Double>(n)
  for (i in 0..<n) {
    Mi = Mi dot Mlist[i]
    AiT[i] = (Adjoint(TransInv(Mi)) dot SlistT[i].transpose()).transpose()
    AdTi[i] = Adjoint(MatrixExp6(VecTose3(AiT[i].transpose() * -thetalist[i])) dot TransInv(Mlist[i]))
    ViT[i + 1] = ((AdTi[i]!! dot ViT[i].transpose()) + AiT[i].transpose() * dthetalist[i]).transpose()
    VdiT[i + 1] = ((AdTi[i]!! dot VdiT[i].transpose()) + AiT[i].transpose() * ddthetalist[i] + (ad(ViT[i + 1].transpose()) dot AiT[i].transpose()) * dthetalist[i]).transpose()
  }
  for (i in (n - 1) downTo 0) {
    Fi = (AdTi[i + 1]!!.transpose() dot Fi) + (Glist[i] dot VdiT[i + 1].transpose()) - (ad(ViT[i + 1].transpose()).transpose() dot (Glist[i] dot ViT[i + 1].transpose()))
    taulist[i] = Fi.transpose() dot AiT[i].transpose()
  }
  return taulist
}

fun MassMatrix(
  thetalist: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
): MultiArray<Double, D2> {
  val n = thetalist.size
  val M = mk.zeros<Double>(n, n)
  for (i in 0..<n) {
    val ddthetalist = mk.zeros<Double>(n)
    ddthetalist[i] = 1.0
    M[i] = InverseDynamics(
      thetalist,
      mk.zeros<Double>(n),
      ddthetalist,
      mk.zeros<Double>(3),
      mk.zeros<Double>(6),
      Mlist,
      Glist,
      Slist
    )
  }
  return M.transpose()
}

fun VelQuadraticForces(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
) = InverseDynamics(
  thetalist,
  dthetalist,
  mk.zeros<Double>(thetalist.size),
  mk.zeros<Double>(3),
  mk.zeros<Double>(6),
  Mlist,
  Glist,
  Slist
)

fun GravityForces(
  thetalist: MultiArray<Double, D1>,
  g: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
) = InverseDynamics(
  thetalist,
  mk.zeros<Double>(thetalist.size),
  mk.zeros<Double>(thetalist.size),
  g,
  mk.zeros<Double>(6),
  Mlist,
  Glist,
  Slist
)

fun EndEffectorForces(
  thetalist: MultiArray<Double, D1>,
  Ftip: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
) = InverseDynamics(
  thetalist,
  mk.zeros<Double>(thetalist.size),
  mk.zeros<Double>(thetalist.size),
  mk.zeros<Double>(3),
  Ftip,
  Mlist,
  Glist,
  Slist
)

fun ForwardDynamics(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  taulist: MultiArray<Double, D1>,
  g: MultiArray<Double, D1>,
  Ftip: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
): MultiArray<Double, D1> {
  return mk.linalg.inv(MassMatrix(thetalist, Mlist, Glist, Slist)) dot (
    taulist -
      VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist) -
      GravityForces(thetalist, g, Mlist, Glist, Slist) -
      EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
    )
}

fun EulerStep(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  ddthetalist: MultiArray<Double, D1>,
  dt: Double
): Pair<MultiArray<Double, D1>, MultiArray<Double, D1>> =
  thetalist + dt * dthetalist to dthetalist + dt*ddthetalist

fun InverseDynamicsTrajectory(
  thetamat: MultiArray<Double, D2>,
  dthetamat: MultiArray<Double, D2>,
  ddthetamat: MultiArray<Double, D2>,
  g: MultiArray<Double, D1>,
  Ftipmat: MultiArray<Double, D2>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>
): MultiArray<Double, D2> {
  val n = thetamat.shape[0]
  val taumat = mk.zeros<Double>(n, thetamat.shape[1])
  for (i in 0..<n) {
    taumat[i] = InverseDynamics(
      thetamat[i],
      dthetamat[i],
      ddthetamat[i],
      g,
      Ftipmat[i],
      Mlist,
      Glist,
      Slist
    )
  }
  return taumat
}

fun ForwardDynamicsTrajectory(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  taumat: MultiArray<Double, D2>,
  g: MultiArray<Double, D1>,
  Ftipmat: MultiArray<Double, D2>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>,
  dt: Double,
  intRes: Int
): Pair<MultiArray<Double, D2>, MultiArray<Double, D2>> {
  val thetamat = mk.ndarray(taumat.toListD2()) // mutable copy
  val dthetamat = thetamat.copy()
  thetamat[0] = thetalist.copy()
  dthetamat[0] = dthetalist.copy()
  var _thetalist = thetalist.copy()
  var _dthetalist = dthetalist.copy()
  for (i in 0..<taumat.shape[0] - 1) {
    for (j in 0..<intRes) {
      val ddthetalist = ForwardDynamics(
        thetalist = _thetalist,
        dthetalist = _dthetalist,
        taulist = taumat[i],
        g = g,
        Ftip = Ftipmat[i],
        Mlist = Mlist,
        Glist = Glist,
        Slist = Slist
      )
      EulerStep(_thetalist, _dthetalist, ddthetalist, dt/intRes).also {
        _thetalist = it.first
        _dthetalist = it.second
      }
    }
    thetamat[i+1] = _thetalist.copy()
    dthetamat[i+1] = _dthetalist.copy()
  }
  return thetamat to dthetamat
}

// *** CHAPTER 9: TRAJECTORY GENERATION ***

fun CubicTimeScaling(Tf: Double, t: Double) = 3 * (1.0 * t / Tf).pow(2) - 2 * (1.0 * t / Tf).pow(3)

fun QuinticTimeScaling(Tf: Double, t: Double) = 10 * (1.0 * t / Tf).pow(3) - 15 * (1.0 * t / Tf).pow(4) + 6 * (1.0 * t / Tf).pow(5)

fun JointTrajectory(thetastart: MultiArray<Double, D1>, thetaend: MultiArray<Double, D1>, Tf: Double, N: Int, method: Int): MultiArray<Double, D2> {
  val timegap = Tf / (N - 1)
  val traj = mk.zeros<Double>(N, thetastart.size)
  for (i in 0..<N) {
    val s = if (method == 3) CubicTimeScaling(Tf, timegap * i)
    else QuinticTimeScaling(Tf, timegap * i)
    traj[i] = s * thetaend + (1 - s) * thetastart
  }
  return traj
}

fun ScrewTrajectory(
  Xstart: MultiArray<Double, D2>,
  Xend: MultiArray<Double, D2>,
  Tf: Double,
  N: Int,
  method: Int
): List<MultiArray<Double, D2>> {
  val timegap = Tf / (N - 1)
  val traj = mutableListOf<MultiArray<Double, D2>>()
  for (i in 0 ..< N) {
    val s = if (method == 3) CubicTimeScaling(Tf, timegap * i)
    else QuinticTimeScaling(Tf, timegap * i)
    traj += Xstart dot MatrixExp6(MatrixLog6(TransInv(Xstart) dot Xend) * s)
  }
  return traj
}

fun CartesianTrajectory(
  Xstart: MultiArray<Double, D2>,
  Xend: MultiArray<Double, D2>,
  Tf: Double,
  N: Int,
  method: Int
): List<MultiArray<Double, D2>> {
  val timegap = Tf / (N - 1)
  val traj = mutableListOf<MultiArray<Double, D2>>()
  val (Rstart, pstart) = TransToRp(Xstart)
  val (Rend, pend) = TransToRp(Xend)
  for (i in 0 ..< N) {
    val s = if (method == 3) CubicTimeScaling(Tf, timegap * i)
    else QuinticTimeScaling(Tf, timegap * i)
    traj += (Rstart dot MatrixExp3(MatrixLog3(Rstart.transpose() dot Rend) * s))
      .cat(s * pend.reshape(3, 1) + (1.0 - s) * pstart.reshape(3, 1), axis=1)
      .cat(mk.ndarray(mk[mk[0.0, 0.0, 0.0, 1.0]]))
  }
  return traj
}

fun ComputedTorque(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  eint: MultiArray<Double, D1>,
  g: MultiArray<Double, D1>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>,
  thetalistd: MultiArray<Double, D1>,
  dthetalistd: MultiArray<Double, D1>,
  ddthetalistd: MultiArray<Double, D1>,
  Kp: Double,
  Ki: Double,
  Kd: Double
): MultiArray<Double, D1> {
  val e = thetalistd - thetalist
  return (MassMatrix(thetalist, Mlist, Glist, Slist) dot (Kp*e + Ki*(eint + e) + Kd*(dthetalistd - dthetalist))) +
    InverseDynamics(thetalist, dthetalist, ddthetalistd, g, mk.zeros<Double>(6), Mlist, Glist, Slist)
}

fun SimulateControl(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  g: MultiArray<Double, D1>,
  Ftipmat: MultiArray<Double, D2>,
  Mlist: MultiArray<Double, D3>,
  Glist: MultiArray<Double, D3>,
  Slist: MultiArray<Double, D2>,
  thetamatd: MultiArray<Double, D2>,
  dthetamatd: MultiArray<Double, D2>,
  ddthetamatd: MultiArray<Double, D2>,
  gtilde: MultiArray<Double, D1>,
  Mtildelist: MultiArray<Double, D3>,
  Gtildelist: MultiArray<Double, D3>,
  Kp: Double,
  Ki: Double,
  Kd: Double,
  dt: Double,
  intRes: Int
): Pair<NDArray<Double, D2>, D2Array<Double>> {
  val (n, m) = thetamatd.shape
  val thetamat = mk.zeros<Double>(n, m)
  val taumat = thetamat.copy()
  val eint = mk.zeros<Double>(thetalist.size)
  var thetacurrent = thetalist.copy()
  var dthetacurrent = dthetalist.copy()
  for (i in 0..<n) {
    val taulist = ComputedTorque(
      thetacurrent,
      dthetacurrent,
      eint,
      gtilde,
      Mtildelist,
      Gtildelist,
      Slist,
      thetamatd[i],
      dthetamatd[i],
      ddthetamatd[i],
      Kp,
      Ki,
      Kd
    )
    for (j in 0..<intRes) {
      val ddthetalist = ForwardDynamics(
        thetalist = thetacurrent,
        dthetalist = dthetacurrent,
        taulist = taulist,
        g = g,
        Ftip = Ftipmat[i],
        Mlist = Mlist,
        Glist = Glist,
        Slist = Slist
      )
      EulerStep(thetacurrent, dthetacurrent, ddthetalist, dt/intRes).also {
        thetacurrent = it.first
        dthetacurrent = it.second
      }
    }
    taumat[i] = taulist
    thetamat[i] = thetacurrent
    eint += dt*(thetamatd[i] - thetacurrent)
  }
  return taumat to thetamat
}
