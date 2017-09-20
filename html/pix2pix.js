// -*- coding: utf-8 -*-

var model = null;
var runner = null;
var sheet = null;
var idx = 0;
var SIZE = 256;

function callback_progressbar(current, total) {
    var pct = Math.round(current / total * 100);
    p.p = pct;
    p.update();
    // console.log("test", current, total);
}

// refer http://cya.sakura.ne.jp/java/loading.htm
function display_inference() {
    var inf = document.querySelector('#calc');
    var myY = (window.innerHeight !== undefined) ? window.innerHeight : document.body.clientHeight;
    var myX = (window.innerWidth !== undefined) ? window.innerWidth : document.body.clientWidth;
    inf.style.top = (myY/2-70 < 0) ? 0 : myY/2-70;
    inf.style.left = (myX/2-300 <0) ? 0 : myX/2-300;
    inf.style.visibility = "visible";
    return inf;
}

function hide_inference() {
    var inf = document.querySelector('#calc');
    inf.style.visibility = "hidden";
}

// ref: https://www.nishishi.com/javascript-tips/setinterval-passage.html

var sec = 0;
function showPassage() {
    sec ++;
    var msg = sec + "sec";
    document.getElementById("sec").innerText = msg;
}

function startTimer() {
    sec = 0;
    PassageId = setInterval('showPassage()', 1000);
    document.getElementById("inference").style.disabled = true;
    document.getElementById("sec").innerText = "";
}

function stopTimer() {
    clearInterval(PassageId);
    document.getElementById("inference").style.disabled = false;
}

window.onload = function() {
    sheet = new Image();
    sheet.src = "sheet.jpg";
    sheet.onload = () => {};
    model = WebDNN.load("../out", {progressCallback: callback_progressbar}).then(function (r) {
    	var stat = document.querySelector("#status");
    	stat.textContent = "Loaded.";
	runner = r;
	var v = document.getElementById("prog-bar");
	v.style.visiblility = "hidden";
    });
};

function set_image() {
    var y_offset = idx * SIZE;
    var inp = document.querySelector("#input");
    var ctx = inp.getContext("2d");
    ctx.drawImage(sheet, 0, y_offset, SIZE, SIZE, 0, 0, SIZE, SIZE);
    var out = document.querySelector("#output");
    ctx = out.getContext("2d");
    // ctx.drawImage(sheet, SIZE, y_offset, SIZE, SIZE, 0, 0, SIZE, SIZE);
    idx = (idx + 1) % (sheet.height / SIZE);
}

function start() {
    console.log("start");
    var img = document.querySelector("#input"); // #edge
    WebDNN.Image.getImageArray(img, { scale: [255, 255, 255], order: WebDNN.Image.Order.CHW}).then(function (img) {
	console.log("runner called");
	display_inference();
	startTimer();
	var x = runner.getInputViews()[0];
	var y = runner.getOutputViews()[0];
	// console.log(img)
	x.set(img);
	runner.run().then(function () {
	    console.log("output");
	    var ret = y.toActual();
	    var cvs = document.querySelector("#output");
	    var clip = [];
	    // console.log(ret);
	    WebDNN.Image.setImageArrayToCanvas(ret, 256, 256, cvs, {
		scale: [255, 255, 255],
		bias: [0, 0, 0],
		color: WebDNN.Image.Color.RGB,
		order: WebDNN.Image.Order.CHW
	    });
	    stopTimer();
	    hide_inference();
	});
    });
}

