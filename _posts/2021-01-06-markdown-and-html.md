---
layout : post
title : Markdown Grammar
tags : [Markdown]
author : Nayeong Kim
comments : True
sitemap :
  changefreq : daily
  priority : 1.0
---

<br>
<h2>마크다운 문법 by Nayeong Kim</h2>
<br>

<h2>1. HTML headings</h2>
{% highlight html %}
<h1>This is heading 1</h1>
<h2>This is heading 2</h2>
<h3>This is heading 3</h3>
<h4>This is heading 4</h4>
<h5>This is heading 5</h5>
<h6>This is heading 6</h6>
{% endhighlight %}
<h1>This is heading 1</h1>
<h2>This is heading 2</h2>
<h3>This is heading 3</h3>
<h4>This is heading 4</h4>
<h5>This is heading 5</h5>
<h6>This is heading 6</h6>

<br>

<h2>2. bold text</h2>
{% highlight html %}
<p>This is normal text - <b>and this is bold text</b>.</p>
{% endhighlight %}
<p>This is normal text - <b>and this is bold text</b>.</p>

<br>

<h2>3. list</h2>
<h3>a. unordered list</h3>
{% highlight html %}
- Coffee
- Tea
- Milk
{% endhighlight %}
- Coffee
- Tea
- Milk

<h3>b. ordered list</h3>
{% highlight html %}
1. Coffee
2. Tea
3. Milk
{% endhighlight %}
1. Coffee
2. Tea
3. Milk

<br>

<h2>4. hyperlink</h2>
{% highlight html %}
[heeseok's blog](https://heeseok-jeong.github.io)
{% endhighlight %}
[heeseok's blog](https://heeseok-jeong.github.io)

<br>

<h2>5. image</h2>
Try using `.width-30`, `.width-40`, `.width-50`, `.width-60`, `.width-70` and `.width-80` class! You can easily change the image width.

{% highlight html %}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg)
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-30}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-50}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-80}
{% endhighlight %}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg)
<p></p>
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-30}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-50}
![sample image]({{ site.baseurl }}/assets/img/koreaSunset.jpg){: .width-80}
<br>

<h2>5. table</h2>
{% highlight html %}
| Header 1  | Header 2 | Header 3 |
| :------- | :-------: | -------: |
| Content 1  | Content 2 | Content 3 |
| Content 1  | Content 2 | Content 3 |
{% endhighlight %}
| Header 1  | Header 2 | Header 3 |
| :------- | :-------: | -------: |
| Content 1 | Content 2 | Content 3 |
| Content 1 | Content 2 | Content 3 |
