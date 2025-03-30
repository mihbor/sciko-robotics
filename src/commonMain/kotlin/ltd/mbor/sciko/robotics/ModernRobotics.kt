package ltd.mbor.sciko.robotics

import ltd.mbor.sciko.linalg.*
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


/**
 * Determines whether a scalar is small enough to be treated as zero
 * @param z A scalar input to check
 * @return True if z is close to zero, false otherwise
 */
fun NearZero(z: Double) = abs(z) < 1e-6

/**
 * Normalizes a vector
 * @param V A vector
 * @return A unit vector pointing in the same direction as z
 */
fun Normalize(V: MultiArray<Double, D1>): MultiArray<Double, D1> {
  return V / V.norm()
}

// *** CHAPTER 3: RIGID-BODY MOTIONS ***

/**
 * Inverts a rotation matrix
 * @param R A rotation matrix
 * @return The inverse of R
 */
fun RotInv(R: MultiArray<Double, D2>) = R.transpose()

/**
 * Converts a 3-vector to an so(3) representation
 * @param omg A 3-vector
 * @return The skew-symmetric matrix representation of omg
 */
fun VecToso3(omg: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return mk.ndarray(mk[
    mk[0.0    , -omg[2],  omg[1]],
    mk[omg[2] ,     0.0, -omg[0]],
    mk[-omg[1],  omg[0],    0.0]])
}

/**
 * Converts an so(3) representation to a 3-vector
 * @param so3mat A 3x3 skew-symmetric matrix
 * @return The 3-vector corresponding to so3mat
 */
fun so3ToVec(so3mat: MultiArray<Double, D2>): MultiArray<Double, D1> {
  return mk.ndarray(mk[so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])
}

/**
 * Converts a 3-vector of exponential coordinates for rotation into axis-angle form
 * @param expc3 A 3-vector of exponential coordinates for rotation
 * @return A unit rotation axis and the corresponding rotation angle
 */
fun AxisAng3(expc3: MultiArray<Double, D1>): Pair<MultiArray<Double, D1>, Double> {
  return Normalize(expc3) to expc3.norm()
}

/**
 * Computes the matrix exponential of a matrix in so(3)
 * @param so3mat A 3x3 skew-symmetric matrix
 * @return The matrix exponential of so3mat
 */
fun MatrixExp3(so3mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val omgtheta = so3ToVec(so3mat)
  return if (NearZero(omgtheta.norm())) mk.identity<Double>(3)
  else {
    val (_, theta) = AxisAng3(omgtheta)
    val omgmat = so3mat / theta
    mk.identity<Double>(3) + sin(theta) * omgmat + (1 - cos(theta)) * (omgmat dot omgmat)
  }
}

/**
 * Computes the matrix logarithm of a rotation matrix
 * @param R A 3x3 rotation matrix
 * @return The matrix logarithm of R
 */
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

/**
 * Converts a rotation matrix and a position vector into homogeneous transformation matrix
 * @param R A 3x3 rotation matrix
 * @param p A 3-vector
 * @return A homogeneous transformation matrix corresponding to the inputs
 */
fun RpToTrans(R: MultiArray<Double, D2>, p: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return R.cat(p.reshape(3, 1), axis=1).cat(mk.ndarray(mk[mk[0.0, 0.0, 0.0, 1.0]]), axis=0)
}

/**
 * Converts a homogeneous transformation matrix into a rotation matrix and position vector
 * @param T A homogeneous transformation matrix
 * @return The corresponding rotation matrix and position vector
 */
fun TransToRp(T: MultiArray<Double, D2>): Pair<MultiArray<Double, D2>, MultiArray<Double, D1>> {
  return T[0..2, 0..2] to T[0..2, 3]
}

/**
 * Inverts a homogeneous transformation matrix
 *
 * Uses the structure of transformation matrices to avoid taking a matrix inverse, for efficiency.
 * @param T A homogeneous transformation matrix
 * @return The inverse of T
 */
fun TransInv(T: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (R, p) = TransToRp(T)
  val Rt = R.transpose()
  return RpToTrans(Rt, -Rt dot p)
}

/**
 * Converts a spatial velocity vector into a 4x4 matrix in se3
 * @param V A 6-vector representing a spatial velocity
 * @return The 4x4 se3 representation of V
 */
fun VecTose3(V: MultiArray<Double, D1>): MultiArray<Double, D2> {
  return VecToso3(V[0..2])
    .cat(V[3..5].reshape(3,1), axis=1)
    .cat(mk.zeros(1,4))
}

/**
 * Converts an se3 matrix into a spatial velocity vector
 * @param se3mat A 4x4 matrix in se3
 * @return The spatial velocity 6-vector corresponding to se3mat
 */
fun se3ToVec(se3mat: MultiArray<Double, D2>): MultiArray<Double, D1> {
  return mk.ndarray(mk[
    se3mat[2, 1], se3mat[0, 2], se3mat[1, 0],
    se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]
  ])
}

/**
 * Computes the adjoint representation of a homogeneous transformation matrix
 * @param T A homogeneous transformation matrix
 * @return The 6x6 adjoint representation [AdT] of T
 */
fun Adjoint(T: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val (R, p) = TransToRp(T)
  return R.cat(mk.zeros(3, 3), axis=1).cat((VecToso3(p) dot R).cat(R, axis=1), axis=0)
}

/**
 * Takes a parametric description of a screw axis and converts it to a normalized screw axis
 * @param q A point lying on the screw axis
 * @param s A unit vector in the direction of the screw axis
 * @param h The pitch of the screw axis
 * @return A normalized screw axis described by the inputs
 */
fun ScrewToAxis(q: MultiArray<Double, D1>, s: MultiArray<Double, D1>, h: Double): MultiArray<Double, D1> {
  return s.cat((VecToso3(q) dot s) + h * s, axis=0)
}

/**
 * Converts a 6-vector of exponential coordinates into screw axis-angle form
 * @param expc6 A 6-vector of exponential coordinates for rigid-body motion - S*theta
 * @return The distance traveled along/about S
 */
fun AxisAng6(expc6: MultiArray<Double, D1>): Pair<MultiArray<Double, D1>, Double> {
  var theta = expc6[0..2].norm()
  if (NearZero(theta))
    theta = expc6[3..5].norm()
  return (expc6 / theta) to theta
}

/**
 * Computes the matrix exponential of an se3 representation of exponential coordinates
 * @param se3mat A matrix in se3
 * @return The matrix exponential of se3mat
 */
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

/**
 * Computes the matrix logarithm of a homogeneous transformation matrix
 * @param T A matrix in SE3
 * @return The matrix logarithm of T
 */
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

/**
 * Returns a projection of mat into SO(3)
 *
 * Projects a matrix mat to the closest matrix in SO(3) using singular-value decomposition
 * (see http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
 * This function is only appropriate for matrices close to SO(3).
 * @param mat A matrix near SO(3) to project to SO(3)
 * @return The closest matrix to R that is in SO(3)
 */
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

/**
 * Returns a projection of mat into SE(3)
 *
 * Projects a matrix mat to the closest matrix in SE(3) using singular-value decomposition
 * (see http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
 * This function is only appropriate for matrices close to SE(3).
 * @param mat A 4x4 matrix to project to SE(3)
 * @return The closest matrix to T that is in SE(3)
 */
fun ProjectToSE3(mat: MultiArray<Double, D2>): MultiArray<Double, D2> {
  return RpToTrans(ProjectToSO3(mat[0..<3, 0..<3]), mat[0..<3, 3])
}

/**
 * Returns the Frobenius norm to describe the distance of mat from the SO(3) manifold
 *
 * Computes the distance from mat to the SO(3) manifold using the following method:
 * If det(mat) <= 0, return a large number.
 * If det(mat) > 0, return norm(mat^T.mat - I).
 * @param mat A 3x3 matrix
 * @return A quantity describing the distance of mat from the SO(3) manifold
 */
fun DistanceToSO3(mat: MultiArray<Double, D2>): Double {
  return if (mk.linalg.det(mat) > 0) mk.linalg.norm((mat.transpose() dot mat) - mk.identity(3))
  else 1e9
}

/**
 * Returns the Frobenius norm to describe the distance of mat from the SE(3) manifold
 *
 * Computes the distance from mat to the SE(3) manifold using the following method:
 * Compute the determinant of matR, the top 3x3 submatrix of mat.
 * If det(matR) <= 0, return a large number.
 * If det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T.matR,
 * and set the first three entries of the fourth column of mat to zero. Then return norm(mat - I).
 * @param mat A 4x4 matrix
 * @return A quantity describing the distance of mat from the SE(3) manifold
 */
fun DistanceToSE3(mat: MultiArray<Double, D2>): Double {
  val matR = mat[0..<3, 0..<3]
  return if (mk.linalg.det(matR) > 0)
    mk.linalg.norm(
      (matR.transpose() dot matR).cat(mk.zeros<Double>(3, 1), axis=1)
        .cat(mat[3].reshape(1, 4)) - mk.identity(4)
    )
  else 1e9
}

/**
 * Returns true if mat is close to or on the manifold SO(3)
 *
 * Computes the distance d from mat to the SO(3) manifold using the following method:
 * If det(mat) <= 0, d = a large number.
 * If det(mat) > 0, d = norm(mat^T.mat - I).
 * If d is close to zero, return true. Otherwise, return false.
 * @param mat A 3x3 matrix
 * @return True if mat is very close to SO(3), false otherwise
 */
fun TestIfSO3(mat: MultiArray<Double, D2>): Boolean {
  return abs(DistanceToSO3(mat)) < 1e-3
}

/**
 * Returns true if mat is close to or on the manifold SE(3)
 *
 * Computes the distance d from mat to the SE(3) manifold using the following method:
 * Compute the determinant of the top 3x3 submatrix of mat.
 * If det(mat) <= 0, d = a large number.
 * If det(mat) > 0, replace the top 3x3 submatrix of mat with mat^T.mat, and
 * set the first three entries of the fourth column of mat to zero.
 * Then d = norm(T - I).
 * If d is close to zero, return true. Otherwise, return false.
 * @param mat A 4x4 matrix
 * @return True if mat is very close to SE(3), false otherwise
 */
fun TestIfSE3(mat: MultiArray<Double, D2>): Boolean {
  return abs(DistanceToSE3(mat)) < 1e-3
}

// *** CHAPTER 4: FORWARD KINEMATICS ***

/**
 * Computes forward kinematics in the body frame for an open chain robot
 * @param M The home configuration (position and orientation) of the end-effector
 * @param Blist The joint screw axes in the end-effector frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param thetalist A list of joint coordinates
 * @return A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates
 * (i.t.o Body Frame)
 */
fun FKinBody(M: MultiArray<Double, D2>, Blist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  var T = M
  val BlistT = Blist.transpose()
  for (i in 0..<thetalist.size) {
    T = T dot MatrixExp6(VecTose3(BlistT[i].transpose()  * thetalist[i]))
  }
  return T
}

/**
 * Computes forward kinematics in the space frame for an open chain robot
 * @param M The home configuration (position and orientation) of the end-effector
 * @param Slist The joint screw axes in the space frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param thetalist A list of joint coordinates
 * @return A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates
 * (i.t.o Space Frame)
 */
fun FKinSpace(M: MultiArray<Double, D2>, Slist: MultiArray<Double, D2>, thetalist: MultiArray<Double, D1>): MultiArray<Double, D2> {
  var T = M
  val SlistT = Slist.transpose()
  for (i in thetalist.size-1 downTo 0) {
    T = MatrixExp6(VecTose3(SlistT[i].transpose() * thetalist[i])) dot T
  }
  return T
}

// *** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***

/**
 * Computes the body Jacobian for an open chain robot
 * @param Blist The joint screw axes in the end-effector frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param thetalist A list of joint coordinates
 * @return The body Jacobian corresponding to the inputs (6xn real numbers)
 */
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

/**
 * Computes the space Jacobian for an open chain robot
 * @param Slist The joint screw axes in the space frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param thetalist A list of joint coordinates
 * @return The space Jacobian corresponding to the inputs (6xn real numbers)
 */
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

// *** CHAPTER 6: INVERSE KINEMATICS ***

/**
 * Computes inverse kinematics in the body frame for an open chain robot
 *
 * Uses an iterative Newton-Raphson root-finding method.
 * @param Blist The joint screw axes in the end-effector frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param M The home configuration of the end-effector
 * @param T The desired end-effector configuration Tsd
 * @param thetalist0 An initial guess of joint angles that are close to satisfying Tsd
 * @param eomg  A small positive tolerance on the end-effector orientation error.
 * The returned joint angles must give an end-effector orientation error less than eomg
 * @param ev A small positive tolerance on the end-effector linear position error.
 * The returned joint angles must give an end-effector position error less than ev
 * @param maxiterations The maximum number of iterations before the algorithm is terminated
 * @return Joint angles that achieve T within the specified tolerances and a Boolean value
 * where true means that the function found a solution
 * and false means that it ran through the set number of maximum iterations
 * without finding a solution within the tolerances eomg and ev.
 */
fun IKinBody(
  Blist: MultiArray<Double, D2>,
  M: MultiArray<Double, D2>,
  T: MultiArray<Double, D2>,
  thetalist0: MultiArray<Double, D1>,
  eomg: Double,
  ev: Double,
  maxiterations: Int = 20
): Pair<MultiArray<Double, D1>, Boolean> {

  var thetalist = thetalist0
  var i = 0
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

/**
 * Computes inverse kinematics in the space frame for an open chain robot
 *
 * Uses an iterative Newton-Raphson root-finding method.
 * @param Slist The joint screw axes in the space frame when the manipulator is at the home position,
 * in the format of a matrix with axes as the columns
 * @param M The home configuration of the end-effector
 * @param T The desired end-effector configuration Tsd
 * @param thetalist0 An initial guess of joint angles that are close to satisfying Tsd
 * @param eomg A small positive tolerance on the end-effector orientation error.
 * The returned joint angles must give an end-effector orientation error less than eomg
 * @param ev A small positive tolerance on the end-effector linear position error.
 * The returned joint angles must give an end-effector position error less than ev
 * @param maxiterations The maximum number of iterations before the algorithm is terminated
 * @return Joint angles that achieve T within the specified tolerances and a Boolean value
 * where true means that the function found a solution
 * and false means that it ran through the set number of maximum iterations
 * without finding a solution within the tolerances eomg and ev.
 */
fun IKinSpace(
  Slist: MultiArray<Double, D2>,
  M: MultiArray<Double, D2>,
  T: MultiArray<Double, D2>,
  thetalist0: MultiArray<Double, D1>,
  eomg: Double,
  ev: Double,
  maxiterations: Int = 20
): Pair<MultiArray<Double, D1>, Boolean> {

  var thetalist = thetalist0
  var i = 0
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

/**
 * Calculate the 6x6 matrix [adV] of the given 6-vector
 *
 * Used to calculate the Lie bracket [V1, V2] = [adV1]V2
 * @param V A 6-vector spatial velocity
 * @return The corresponding 6x6 matrix [adV]
 */
fun ad(V: MultiArray<Double, D1>): MultiArray<Double, D2> {
  val omgmat = VecToso3(mk.ndarray(mk[V[0], V[1], V[2]]))
  return omgmat.cat(mk.zeros(3, 3), axis=1).cat(
    VecToso3(mk.ndarray(mk[V[3], V[4], V[5]])).cat(omgmat, axis=1), axis=0)
}

/**
 * Computes inverse dynamics in the space frame for an open chain robot
 *
 * This function uses forward-backward Newton-Euler iterations to solve the equation:
 * taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) + g(thetalist) + Jtr(thetalist)Ftip
 * @param thetalist n-vector of joint variables
 * @param dthetalist n-vector of joint rates
 * @param ddthetalist n-vector of joint accelerations
 * @param g Gravity vector g
 * @param Ftip Spatial force applied by the end-effector expressed in frame {n+1}
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The n-vector of required joint forces/torques
 */
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

/**
 * Computes the mass matrix of an open chain robot based on the given configuration
 *
 * This function calls InverseDynamics n times, each time passing a ddthetalist vector
 * with a single element equal to one and all other inputs set to zero.
 * Each call of InverseDynamics generates a single column, and these columns
 * are assembled to create the inertia matrix.
 * @param thetalist A list of joint variables
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The numerical inertia matrix M(thetalist) of an n-joint serial chain at the given configuration thetalist
 */
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

/**
 * Computes the Coriolis and centripetal terms in the inverse dynamics of an open chain robot
 *
 * This function calls InverseDynamics with g = 0, Ftip = 0, and ddthetalist = 0.
 * @param thetalist A list of joint variables
 * @param dthetalist A list of joint rates
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The vector c(thetalist,dthetalist) of Coriolis and centripetal terms for a given thetalist and dthetalist.
 */
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

/**
 * Computes the joint forces/torques an open chain robot requires to overcome gravity at its configuration
 *
 * This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and ddthetalist = 0.
 * @param thetalist A list of joint variables
 * @param g 3-vector for gravitational acceleration
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The joint forces/torques required to overcome gravity at thetalist
 */
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

/**
 * Computes the joint forces/torques an open chain robot requires only to create the end-effector force Ftip
 *
 * This function calls InverseDynamics with g = 0, dthetalist = 0, and ddthetalist = 0.
 * @param thetalist A list of joint variables
 * @param Ftip The spatial force applied by the end-effector expressed in frame {n+1}
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The joint forces/torques required only to create the end-effector force Ftip
 */
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

/**
 * Computes forward dynamics in the space frame for an open chain robot
 *
 * This function computes ddthetalist by solving:
 * Mlist(thetalist) * ddthetalist = taulist - c(thetalist,dthetalist) - g(thetalist) - Jtr(thetalist) * Ftip
 * @param thetalist A list of joint variables
 * @param dthetalist A list of joint rates
 * @param taulist An n-vector of joint forces/torques
 * @param g Gravity vector g
 * @param Ftip Spatial force applied by the end-effector expressed in frame {n+1}
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The resulting joint accelerations
 */
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

/**
 * Compute the joint angles and velocities at the next timestep using from here first order Euler integration
 * @param thetalist n-vector of joint variables
 * @param dthetalist n-vector of joint rates
 * @param ddthetalist n-vector of joint accelerations
 * @param dt The timestep delta t
 * @return Vector of joint variables after dt from first order Euler integration and
 * vector of joint rates after dt from first order Euler integration
 */
fun EulerStep(
  thetalist: MultiArray<Double, D1>,
  dthetalist: MultiArray<Double, D1>,
  ddthetalist: MultiArray<Double, D1>,
  dt: Double
): Pair<MultiArray<Double, D1>, MultiArray<Double, D1>> =
  thetalist + dt * dthetalist to dthetalist + dt*ddthetalist

/**
 * Calculates the joint forces/torques required to move the serial chain along the given trajectory using inverse dynamics
 * @param thetamat An N x n matrix of robot joint variables
 * @param dthetamat An N x n matrix of robot joint velocities
 * @param ddthetamat An N x n matrix of robot joint accelerations
 * @param g Gravity vector g
 * @param Ftipmat An N x 6 matrix of spatial forces applied by the end-effector
 * (If there are no tip forces, the user should input a zero matrix)
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @return The N x n matrix of joint forces/torques for the specified trajectory,
 * where each of the N rows is the vector of joint forces/torques at each time step
 */
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

/**
 * Simulates the motion of a serial chain given an open-loop history of joint forces/torques
 * @param thetalist n-vector of initial joint variables
 * @param dthetalist n-vector of initial joint rates
 * @param taumat An N x n matrix of joint forces/torques, where each row is the joint effort at any time step
 * @param g Gravity vector g
 * @param Ftipmat An N x 6 matrix of spatial forces applied by the end-effector
 * (If there are no tip forces, the user should input a zero matrix)
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @param dt The timestep between consecutive joint forces/torques
 * @param intRes Integration resolution is the number of times integration (Euler) takes places between each time step.
 * Must be greater than or equal to 1
 * @return The N x n matrix of robot joint angles resulting from the specified joint forces/torques and
 * the N x n matrix of robot joint velocities
 */
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

/**
 * Computes s(t) for a cubic time scaling
 * @param Tf Total time of the motion in seconds from rest to rest
 * @param t The current time t satisfying 0 < t < Tf
 * @return The path parameter s(t) corresponding to a third-order polynomial motion that begins and ends at zero velocity
 */
fun CubicTimeScaling(Tf: Double, t: Double) = 3 * (1.0 * t / Tf).pow(2) - 2 * (1.0 * t / Tf).pow(3)

/**
 * Computes s(t) for a quintic time scaling
 * @param Tf Total time of the motion in seconds from rest to rest
 * @param t The current time t satisfying 0 < t < Tf
 * @return The path parameter s(t) corresponding to a fifth-order polynomial motion
 * that begins and ends at zero velocity and zero acceleration
 */
fun QuinticTimeScaling(Tf: Double, t: Double) = 10 * (1.0 * t / Tf).pow(3) - 15 * (1.0 * t / Tf).pow(4) + 6 * (1.0 * t / Tf).pow(5)

/**
 * Computes a straight-line trajectory in joint space
 * @param thetastart The initial joint variables
 * @param thetaend The final joint variables
 * @param Tf Total time of the motion in seconds from rest to rest
 * @param N The number of points N > 1 (Start and stop) in the discrete representation of the trajectory
 * @param method The time-scaling method, where 3 indicates cubic (third-order polynomial) time scaling
 * and 5 indicates quintic (fifth-order polynomial) time scaling
 * @return A trajectory as an N x n matrix, where each row is an n-vector of joint variables at an instant in time.
 * The first row is thetastart and the Nth row is thetaend.
 * The elapsed time between each row is Tf / (N - 1)
 */
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

/**
 * Computes a trajectory as a list of N SE(3) matrices corresponding to the screw motion about a space screw axis
 * @param Xstart The initial end-effector configuration
 * @param Xend The final end-effector configuration
 * @param Tf Total time of the motion in seconds from rest to rest
 * @param N The number of points N > 1 (Start and stop) in the discrete representation of the trajectory
 * @param method The time-scaling method, where 3 indicates cubic (third-order polynomial) time scaling
 * and 5 indicates quintic (fifth-order polynomial) time scaling
 * @return The discretized trajectory as a list of N matrices in SE(3) separated in time by Tf/(N-1).
 * The first in the list is Xstart and the Nth is Xend
 */
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

/**
 * Computes a trajectory as a list of N SE(3) matrices
 * corresponding to the origin of the end-effector frame following a straight line
 *
 * This function is similar to ScrewTrajectory, except the origin of the end-effector frame follows a straight line,
 * decoupled from the rotational motion.
 * @param Xstart The initial end-effector configuration
 * @param Xend The final end-effector configuration
 * @param Tf Total time of the motion in seconds from rest to rest
 * @param N The number of points N > 1 (Start and stop) in the discrete representation of the trajectory
 * @param method The time-scaling method, where 3 indicates cubic (third-order polynomial) time scaling
 * and 5 indicates quintic (fifth-order polynomial) time scaling
 * @return The discretized trajectory as a list of N matrices in SE(3) separated in time by Tf/(N-1).
 * The first in the list is Xstart and the Nth is Xend
 */
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

// *** CHAPTER 11: ROBOT CONTROL ***

/**
 * Computes the joint control torques at a particular time instant
 * @param thetalist n-vector of joint variables
 * @param dthetalist n-vector of joint rates
 * @param eint n-vector of the time-integral of joint errors
 * @param g Gravity vector g
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @param thetalistd n-vector of reference joint variables
 * @param dthetalistd n-vector of reference joint velocities
 * @param ddthetalistd n-vector of reference joint accelerations
 * @param Kp The feedback proportional gain (identical for each joint)
 * @param Ki The feedback integral gain (identical for each joint)
 * @param Kd The feedback derivative gain (identical for each joint)
 * @return The vector of joint forces/torques computed by the feedback linearizing controller at the current instant
 */
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

/**
 * Simulates the computed torque controller over a given desired trajectory
 * @param thetalist n-vector of initial joint variables
 * @param dthetalist n-vector of initial joint velocities
 * @param g Actual gravity vector g
 * @param Ftipmat An N x 6 matrix of spatial forces applied by the end-effector
 * (If there are no tip forces, the user should input a zero matrix)
 * @param Mlist List of link frames {i} relative to {i-1} at the home position
 * @param Glist Spatial inertia matrices Gi of the links
 * @param Slist Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
 * @param thetamatd An N x n matrix of desired joint variables from the reference trajectory
 * @param dthetamatd An N x n matrix of desired joint velocities
 * @param ddthetamatd An N x n matrix of desired joint accelerations
 * @param gtilde The gravity vector based on the model of the actual robot (actual values given above)
 * @param Mtildelist The link frame locations based on the model of the actual robot (actual values given above)
 * @param Gtildelist The link spatial inertias based on the model of the actual robot (actual values given above)
 * @param Kp The feedback proportional gain (identical for each joint)
 * @param Ki The feedback integral gain (identical for each joint)
 * @param Kd The feedback derivative gain (identical for each joint)
 * @param dt The timestep between points on the reference trajectory
 * @param intRes Integration resolution is the number of times integration (Euler) takes places between each time step.
 * Must be greater than or equal to 1
 * @return An Nxn matrix of the controllers commanded joint forces/torques,
 * where each row of n forces/torques corresponds to a single time instant and
 * an Nxn matrix of actual joint angles
 */
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
