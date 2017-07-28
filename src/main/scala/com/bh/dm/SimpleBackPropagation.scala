package com.bh.d406.bigdata.bhdm.bhdl

import scala.collection.mutable.{ArrayBuffer, Map => MuMap, Set => MuSet}
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable

/**
  * Created by lulin on 2017/5/18.
  */
trait Forward_Backward {
  // define two abstract methods
  def forward : Unit
  def backward : Unit
}

abstract class Node(val inbound_nodes: Array[Node] = Array.empty) extends Forward_Backward{

  val outbound_nodes: ArrayBuffer[Node] = ArrayBuffer[Node]()

  var value: DenseMatrix[Double] = _

  var gradients: MuMap[Node, DenseMatrix[Double]] = MuMap.empty[Node, DenseMatrix[Double]]

  // set this node to be output of received input nodes
  if(inbound_nodes.nonEmpty)
    for(in <- inbound_nodes)
      in.outbound_nodes.append(this)
}





/**
  * An input node contains value of `X`, `w` or `b`
  *
  * 正向传播：从入度边{@param inbound_nodes}收集梯度
  * 反向传播：从出度边{@param outbound_nodes}收集梯度
  *
  */
class Input extends Node {

  override def forward: Unit = {
    println("Input node, forward nothing")
  }

  /**
    * An Input node has no inputs so the gradient (derivative) is zero.
      The key, `self`, is reference to this object.

      Weights and bias may be inputs, so you need to sum
      the gradient from output gradients.
    */
  override def backward: Unit = {
    println("Backward to Input node")
//    gradients.put(this, DenseMatrix(0.0))
//    Weights and bias may be inputs, so you need to sum
//    the gradient from output gradients.
    for(n <- outbound_nodes)
      if(gradients.contains(this))
        gradients(this) += n.gradients(this)  //Elementwise addition, just +
      else
        gradients.put(this, n.gradients(this))
    // 这里 n.gradients(this) 类似于链式法则中的 partial(n) / partial(this)
    // 梯度应该就是一个scalar
  }

}

/**
  * 线性计算结点 value = X dot W + bias
  * @param X input data matrix
  * @param W  weight matrix
  * @param bias  bias vector, maybe vector
  */
class Linear(val X:Node, val W: Node, val bias: Node) extends Node(Array(X, W, bias)) {

  // 计算 W*X+b
  override def forward: Unit = {
    val X = inbound_nodes(0).value
    val W = inbound_nodes(1).value
    val dotValue = X * W
    val b = inbound_nodes(2).value.toDenseVector // using broadcast to add vector b
    this.value = dotValue(*, ::) + b
    println("Linear node forward value is "+ value)
  }

  override def backward: Unit = {
    println("Backward to Linear node")
    //Initialize a partial for each of the inbound_nodes.
    inbound_nodes.foreach{ n =>
      gradients += n -> DenseMatrix.zeros(n.value.rows, n.value.cols)
    }
    //Collect the gradients from outbound_nodes
    for(n <- outbound_nodes) {
      //Get the partial of the cost with respect to this node.
      val grad_cost = n.gradients(this)
      //Set the partial of the loss with respect to this node's inputs.
      gradients(inbound_nodes(0)) +=  grad_cost * inbound_nodes(1).value.t
      //Set the partial of the loss with respect to this node's weights.
      gradients(inbound_nodes(1)) += (inbound_nodes(0).value.t * grad_cost)
      //Set the partial of the loss with respect to this node's bias.
      gradients(inbound_nodes(2)) += sum(grad_cost, Axis._0)
    }
  }

}


class Sigmoid(val node: Node) extends Node(Array(node)) {

  // value of Sigmoid node should be a vector

  override def forward: Unit = {
    val input = inbound_nodes.head.value
    value = exp(input)
    // following code also works
//    input.map{ in =>
//      1.0 / (1.0 + math.exp(-in))
//    }
    println("Sigmoid node forward value is "+value)
  }

  /**
    * {@param node} is the input, to backward means to compute: partial(this) / partial(node)
    * that is denoted by {this.gradients(node)}, but we should collect the gradients from outputs
    * of this
    */
  override def backward: Unit = {
    println("Backward to a Sigmoid node")
    //Initialize a partial for each of the inbound_nodes.
    inbound_nodes.foreach{ n =>
      gradients += n -> DenseMatrix.zeros(n.value.rows, n.value.cols) // zero_like
    }
    //Sum the partial with respect to the input over all the outputs.
    for(n <- outbound_nodes) {
      val grad_cost = n.gradients(this)
      val sigmoid = this.value
      val one_minus_sigmoid = sigmoid.map(v => 1-v)
      val slope = (sigmoid :* one_minus_sigmoid)
      val tmp = slope :* grad_cost
      val predG = gradients(inbound_nodes.head)
      this.gradients(inbound_nodes.head) +=  tmp//slope * grad_cost
    }
  }
}

class MSE(val y: Node, val pred: Node) extends Node(Array(y, pred)) {
  // Calculates the mean squared error.
  override def forward: Unit = {
    // Making both arrays (3,1) insures the result is (3,1) and does
    val flatY = inbound_nodes(0).value//.reshape(*, 1)
    val flatA = inbound_nodes(1).value//.reshape(*, 1)
    val diff = flatA - flatY
    value = DenseMatrix(mean(pow(diff,2)))
    println("MSE node forward value " + value)
  }

  override def backward: Unit = {
    println("Backward from MSE")
    val flatY = inbound_nodes(0).value//.reshape(-1, 1)
    val flatA = inbound_nodes(1).value//.reshape(-1, 1)
    val diff = (flatA - flatY).toDenseMatrix
    val m = inbound_nodes(0).value.rows
    println("MSE the size m of Y is "+m)
    if(gradients.contains(inbound_nodes(0)))
      gradients(inbound_nodes(0)) += diff.map(_ * 2/m)
    else
      gradients.put(inbound_nodes(0), diff.map(_ * 2/m))

    if(gradients.contains(inbound_nodes(1)))
      gradients(inbound_nodes(1)) += diff.map(_ * -2/m)
    else
      gradients.put(inbound_nodes(1),diff.map(_ * -2/m))
  }
}




object NN {


  def main(args: Array[String]): Unit = {

    val X:Node = new Input()
    val W:Node = new Input()
    val b:Node =  new Input()
    val y:Node = new Input()

    val X_ = DenseMatrix((-1.0, -2.0), (-1.0 , -2.0))
    val W_ = new DenseMatrix(2, 1, Array(2.0, 3.0))
    val b_ = DenseMatrix(-3.0)
    val y_ = DenseMatrix(1.0,2.0)

    val f = new Linear(X, W, b)
    val a = new Sigmoid(f)
    val cost = new MSE(y, a)

//println(X_)
//    println(W_)
    println(b_)
    val d = X_ * W_

    val feed_dict = Map[Node, DenseMatrix[Double]](X -> X_, y -> y_, W -> W_, b -> b_)

    val graph = topological_sort(feed_dict)

    forward_and_backward(graph)

    // return the gradients for each Input
    println(feed_dict.keys.head.gradients(feed_dict.keys.head))
//    val gradients = feed_dict.keys.foreach(t => println(t.gradients(t)))

    """Expected output
      |
      |[array([[ -3.34017280e-05,  -5.01025919e-05],
      |       [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
      |       [ 1.9999833]]), array([[  5.01028709e-05],
      |       [  1.00205742e-04]]), array([ -5.01028709e-05])]"""

//    print(gradients)

  }

  /**
    * Sort the nodes in topological order using Kahn's Algorithm.
    *
    * @param feed_dict : A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.
    * @return a list of sorted nodes.
    */
  def topological_sort(feed_dict: Map[Node, DenseMatrix[Double]]): Array[Node] = {
    val input_nodes = feed_dict.keys//.toList //.keySet.reverseIterator

    val G = MuMap[Node, Map[String, MuSet[Node]]]()
    val nodes = ArrayBuffer[Node]()
    nodes.appendAll(input_nodes)
    while(nodes.nonEmpty) {
      val n = nodes.remove(0)
      if(!G.contains(n))
        G.put(n, Map("in"->MuSet.empty[Node], "out"->MuSet.empty[Node]))
      for(m <- n.outbound_nodes) {
        if(!G.contains(m))
          G.put(m, Map("in"->MuSet.empty[Node], "out"->MuSet.empty[Node]))
        G(n)("out").add(m)
        G(m)("in").add(n)
        nodes.append(m)
      }
    }

    val L = new ArrayBuffer[Node]
    val S = mutable.Set[Node]()
    S.++=(input_nodes)
    println(S)
//      mutable.Queue[Node]()

    val q = mutable.Queue[Int]()
    q.enqueue(1)
    q.enqueue(2)
    print(q)
    println(q.dequeue())

    val set = mutable.Set(1,3)
    println(set)

    while(S.nonEmpty) {
      val n = S.last //.dequeue()
      S -= n

      if(n.isInstanceOf[Input])
        n.value = feed_dict(n)

      L.append(n)
      for(m <- n.outbound_nodes) {
        G(n)("out").remove(m)
        G(m)("in").remove(n)
        // if no other incoming edges add to S
        if(G(m)("in").isEmpty)
          S.add(m)
      }
    }

    L.toArray
  }

  def forward_and_backward(graph: Array[Node]) = {
    graph.foreach(_.forward)

    graph.reverseIterator.foreach(_.backward)

    """Input forward
      |Input forward
      |Input forward
      |Input forward
      |Linear forward
      |_sigmoid forward
      |MSE forward
      |MSE backward
      |_sigmoid backward
      |Linear backward
      |Backward to Input
      |Backward to Input
      |Backward to Input
      |Backward to Input"""
  }

}