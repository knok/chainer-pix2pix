// -*- coding: utf-8 -*-

// ref: http://kuroeveryday.blogspot.jp/2016/03/progressbar.html
var Progress = (function () {
    function Progress (p) {
	this.bar = document.querySelectorAll('#prog-bar > .progress-bar')[0];
	this.p = p;
	this.update();
    };
    Progress.prototype.update = function ()  {
	this.bar.style.width = this.p + '%';
    };
    Progress.prototype.countup = function () {
	if (this.p < 100) { this.p += 10; }
	this.update();
    };
    Progress.prototype.countdown = function () {
	if (0 < this.p) { this.p -= 10; }
	this.update();
    };
    return Progress;
}());

var p = new Progress(0);
