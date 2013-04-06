var width = 940;
var height = 500;

function show_graph(graph) {
  var nodes = graph.nodes;
  var links = graph.links;

  var w = width,
      h = height,
      fill = d3.scale.category20();

  var vis = d3.select("#graph")
    .append("svg:svg")
      .attr("width", w)
      .attr("height", h)
      .attr("pointer-events", "all")
    .append("svg:g")
      .call(d3.behavior.zoom().on("zoom", redraw))
    .append("svg:g");
  
  vis.append('svg:rect')
      .attr('x', -100000)
      .attr('y', -100000)
      .attr('width', 200000)
      .attr('height', 200000)
      .attr('fill', 'white');
  
  function redraw() {
    //console.log("here", d3.event.translate, d3.event.scale);
    vis.attr("transform",
        "translate(" + d3.event.translate + ")"
        + " scale(" + d3.event.scale + ")");
  }
  

  var force = d3.layout.force()
      .nodes(nodes)
      .links(links)
      .linkDistance(function (d) { return d.distance ? d.distance : 10; })
      .charge(-1000)
      .size([w, h])
      .start();

  // set link and add svg:line
  var link = vis.selectAll("line.link")
      .data(links)
    .enter()
      .insert("svg:line")
      .attr("class", "link")
      .style("stroke", function(d) {
        return d.stroke ? d.stroke : '#ccc';
      })
      .style("stroke-width", function(d) {
        return d.strokeWidth ? d.strokeWidth : 1;
      })
      .attr("x1", function (d) { return d.source.x; })
      .attr("y1", function (d) { return d.source.y; })
      .attr("x2", function (d) { return d.target.x; })
      .attr("y2", function (d) { return d.target.y; });

  var node = vis.selectAll("g.node")
      .data(nodes)
    .enter()
      .append("svg:g")
      .attr("class", "node")
      .attr("key", function(d) { return d.key; })
      .attr("fixed", function(d) { return d.fixed ? d.fixed : false; })
      .call(force.drag);

  // add text
  node.append("svg:text")
      .attr("class", "nodetext")
      .attr("dx", function (d) { return 4 + Math.sqrt(d.value) * 20.0; })
      .attr("dy", ".35em").text(function (d) { return d.key });

  // add filled circle
  node.append("circle")
      .attr("class", "node")
      .attr("r", function (d) { return Math.sqrt(d.value) * 20.0; })
      .style("fill", function (d) { return fill(d.group); });

  force.on("tick", function () {
      link.attr("x1", function (d) { return d.source.x; })
      .attr("y1", function (d) { return d.source.y; })
      .attr("x2", function (d) { return d.target.x; })
      .attr("y2", function (d) { return d.target.y; });

      node.attr("transform", function (d) {
        return "translate(" + d.x + "," + d.y + ")";
      });
  });
}

function join_graph(a, b) {
  // a == {nodes: …, links: …}
  // それぞれの最初をつなげる
  var new_graph = {};
  new_graph.nodes = [];
  new_graph.links = [];
  
  for (var i = 0; i < a.nodes.length; i++) {
    new_graph.nodes.push(a.nodes[i]);
  }
  for (var i = 0; i < b.nodes.length; i++) {
    new_graph.nodes.push(b.nodes[i]);
  }
  for (var i = 0; i < a.links.length; i++) {
    new_graph.links.push(a.links[i]);
  }
  for (var i = 0; i < a.links.length; i++) {
    var new_link = a.links[i];
    new_link.source += a.nodes.length;
    new_link.target += a.nodes.length;
    new_graph.links.push(new_link);
  }
  return new_graph;
}

// テキストからグラフを生成して表示する
function get_graph(text) {
  $.post("td_attr", {
    "text": text,
    "output": "json",
    "lang": "ja"
  }, function (data) {  
    // nodesとlinksの設定
    var nodes = data.elements.attr;
    nodes.sort(function(a,b){return b.value-a.value});
    var node_keys = []; // ノードキー
    var max_value = nodes[0].value;
    nodes[0].fixed = true;
    nodes[0].x = width / 2;
    nodes[0].y = height / 2;
    
    var g = 0; // グループの設定
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].group = g++ % 20; // 適当にグループを設定
      nodes[i].value /= max_value; // 正規化
      node_keys[i] = nodes[i].key;
    }
    
    var links = [];
    for (var i = 0; i < data.elements.td.length; i++) {
      var td = data.elements.td[i];
      if (td.value > 0.7) {
        links.push({
          "source": node_keys.indexOf(td.source),
          "target": node_keys.indexOf(td.target),
          "value": td.value
        });
      }
    }

    show_graph({nodes:nodes, links:links});
    
    // データの表示
    $("#myTable tbody").empty();
    for (var i = 0; i < data.elements.attr.length; i++) {
      var key = data.elements.attr[i].key;
      var value = data.elements.attr[i].value;
      var tr = $('<tr><td>' + i + '</td><td>' + key + '</td><td>' + value + '</td></tr>');
      tr.click(function() {
        var key = $('td:eq(1)',this).text();
        var circles = d3.select("g.node[key='" + key + "'] circle");
        circles.transition().duration(500).style("opacity", 0.1);
      });
      $("#myTable tbody").append(tr);
    }
        
  });
}


function sample_graph() {
  var nodes = [];
  nodes.push({key:'foo', value:0.1, group:1});
  nodes.push({key:'bar', value:0.4, group:1});
  nodes.push({key:'baz', value:0.5, group:1});

  nodes.push({key:'foo', value:0.4, group:2});
  nodes.push({key:'bar', value:0.2, group:2});
  nodes.push({key:'baz', value:1, group:2});
  
  var links = []
  links.push({source:0, target:1, distance:10});
  links.push({source:2, target:1, distance:10});
  links.push({source:0, target:3, distance:100, stroke:'#ff0', strokeWidth: 5});
  links.push({source:3, target:4});
  links.push({source:5, target:4});
  links.push({source:5, target:3});
  
  show_graph(nodes, links);
}


$(document).ready(function () {

  $('#myModal').modal('show');
  $("#get_graph").click(function () {
    if ($("#graph").children().length > 0) $("#graph").children().remove();
    get_graph($("#text").val());
    $('#myModal').modal('hide')
  });
  $("#toggle_graph").click(function () {
    $("#graph").fadeToggle("fast");
  });
  $("#toggle_panel").click(function () {
    $("#panel").fadeToggle("fast");
  });
  
  sample_graph();
  
});
