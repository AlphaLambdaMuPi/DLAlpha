import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import mpld3
from flask import Flask, render_template, request
from flask.ext.pymongo import PyMongo
from datetime import datetime, timedelta
from bokeh.plotting import *
from bokeh.io import *
from bokeh.io import cursession as curs
from bokeh.embed import *
from bokeh.session import Session
from bokeh.document import Document
app = Flask(__name__)

mongo = PyMongo(app)

@app.route('/post/send_status', methods=['POST'])
def receive_status():
    name = request.form['name']
    status = request.form['status']

    if status == 'new':
        output_server(name)
        p = figure(x_axis_type='datetime')
        p.line([], [], name='Ainl', color='blue', legend='A_in')
        p.line([], [], name='Avall', color='green', legend='A_val')
        p.circle([], [], name='Ainc')
        p.circle([], [], name='Avalc', color='green')
        push()
        cursession().publish()
        docid = curdoc().docid
        plotid = curdoc().ref['id']
        mongo.db.profiles.insert({
            'name': name,
            'type': request.form['type'],
            'status': 'active',
            'time_stamp': request.form['time_stamp'],
            'docid': docid,
            'plotid': plotid,
        })
        return 'OK'
    elif status == 'end':
        mongo.db.profiles.update({
            'name': name
        }, {
            '$set': {'status': 'ended'}
        })
        return 'OK'


@app.route('/post/send_data', methods=['POST'])
def receive_data():
    mongo.db.datas.insert({
        'name': request.form['name'],
        'item': request.form['item'],
        'value': request.form['value'],
        'time_stamp': request.form['time_stamp'],
    })
    name = request.form['name']
    output_server(name)
    cursession().load_document(curdoc())
    if not (curdoc().context.children):
        p = figure(x_axis_type='datetime')
        p.line([], [], name='Ainl')
        p.circle([], [], name='Ainc')
        p.line([], [], name='Avall')
        p.circle([], [], name='Avalc')
        push()
        cursession().publish()
        docid = curdoc().docid
        plotid = curdoc().ref['id']
        mongo.db.profiles.update({
            'name': name
        }, {
            '$set': {'docid': docid, 'plotid': plotid}
        })

    item = request.form['item']
    ds = curdoc().context.children[0].select({"name": item+'c'})
    ds[0].data_source.data['x'].append(datetime.strptime(request.form['time_stamp'], '%Y-%m-%d %H:%M:%S.%f'))
    ds[0].data_source.data['y'].append(float(request.form['value']))
    cursession().store_objects(ds[0])
    ds = curdoc().context.children[0].select({"name": item+'l'})
    ds[0].data_source.data['x'].append(datetime.strptime(request.form['time_stamp'], '%Y-%m-%d %H:%M:%S.%f'))
    ds[0].data_source.data['y'].append(float(request.form['value']))
    cursession().store_objects(ds[0])
    cursession().publish()
    return 'OK'

@app.route('/graph/<name>')
def give_graph(name):
    htmlcode = ''
    #items = mongo.db.datas.find({'name': name}).distinct('item')
    #for i in range(len(items)):
        #datas = mongo.db.datas.find({'name': name, 'item': items[i]}).sort([('time_stamp', -1)]).limit(50)
        #fig, ax = plt.subplots(1, 1)
        #times = []
        #values = []
        #for d in datas:
            #times.append(datetime.strptime(d['time_stamp'], '%Y-%m-%d %H:%M:%S.%f'))
            #values.append(d['value'])
        #ax.plot_date(times, values, 'o-', ms=8, label=items[i])
        #ax.margins(x=.05, y=0.02)
        #ax.grid()
        #ax.legend(loc=8, shadow=True, fancybox=True)
        #fig.autofmt_xdate()
        ##ax.set_xlim(ax.get_xlim()[0]-0.1, ax.get_xlim()[1]+0.1)
        ##ax.set_ylim(ax.get_ylim()[0]-0.1, ax.get_ylim()[1]+0.1)
        #htmlcode += mpld3.fig_to_html(fig)
        #plt.close()
    
    gr = mongo.db.profiles.find_one({'name': name})
    output_server(name)
    cursession().load_document(curdoc())
    htmlcode = autoload_server(curdoc().context.children[0], cursession()).replace('localhost', '140.112.18.227')
    return render_template('graph.html', name=name, htmlcode=htmlcode)

@app.route('/')
def hello():
    #output_server('test')
    #plot = figure()
    #plot.circle([1,2], [3,4])
    #push()
    #cursession().publish()
    #script = autoload_server(plot, cursession(), public=True)
    #return ('<html><head><link rel="stylesheet" '
    #'href="http://cdn.pydata.org/bokeh/release/bokeh-0.8.2.min.css"'
    #' type="text/css"><script type="text/javascript" '
    #'src="http://cdn.pydata.org/bokeh/release/bokeh-0.8.2.min.js">'
    #'</script></head><body>{0}</body></html>').format(script)
    res = list(mongo.db.profiles.find().sort([('time_stamp', -1)]))
    return render_template('index.html', profiles=res)



if __name__ == '__main__':
    app.run()
