package com.bh.dm.wudi
import org.joda.time.DateTime
import scala.io.Source
/**
  * Created by Administrator on 2017/3/3.
  */
object Test {
  def toLongTimestamp(datetime: String): Long = {
    import java.text.SimpleDateFormat
    val format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    val date = format.parse(datetime)
    val d = new DateTime(date)
    d.getMillis()
  }

  def main(args: Array[String]): Unit = {
    val file = Source.fromFile("C:\\Users\\Administrator.admin-PC\\Desktop\\新建文本文档.txt")
    for (line <- file.getLines) {
      val timeString = line.toString
      println(toLongTimestamp(timeString))
    }
    file.close
  }
}
