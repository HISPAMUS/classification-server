import argparse
import flask
from flask import request, jsonify
import logging
import os

from e2e_classifier import E2EClassifier
from image_storage import ImageStorage
from symbol_classifier import SymbolClassifier
from model_manager import ModelManager


app = flask.Flask(__name__)
_logger = logging.getLogger('server')
_storage = None
_symbol_classifier = None
_e2e_classifier = None
_model_manager = None


def message(text):
    return jsonify({ 'message': text })


@app.route('/image', methods=['POST'])
def image_save():
    if request.files.get('image') and request.form.get('id'):
        id = request.form['id']
        request.files['image'].save(_storage.path(id))
        return message(f'Image [{id}] stored'), 200
    else:
        return message('Missing data'), 400


@app.route('/image/<id>', methods=['GET'])
def image_check(id):
    if _storage.exists(id):
        return message(f'Image [{id}] exists'), 200
    else:
        return message(f'Image [{id}] does not exist'), 404


@app.route('/image/<id>', methods=['DELETE'])
def image_delete(id):
    if _storage.exists(id):
        os.remove(_storage.path(id))
        return message(f'Image [{id}] deleted'), 200
    else:
        return message(f'Image [{id}] does not exist'), 404

@app.route('/models', methods=['GET'])
def getAvailableModels():
    modelList = _model_manager.getModelList(request.form['notationType'], request.form['manuscriptType'],request.form.get('project'))
    return message(modelList), 200


@app.route('/image/<id>/symbol', methods=['POST'])
@app.route('/image/<id>/bbox', methods=['POST'])
def symbol_classify(id):

    if not _storage.exists(id):
        return message(f'Image [{id}] does not exist'), 404
    
    try:
        left = int(request.form['left'])
        top = int(request.form['top'])
        right = int(request.form['right'])
        bottom = int(request.form['bottom'])
        n = int(request.form.get('predictions', "1"))
    except ValueError as e:
        return message('Wrong input values'), 400

    try:
        shape_image, position_image = _storage.crop(id, left, top, right, bottom)
    except Exception as e:
        return message('Error cropping image'), 400
    
    try:
        model = _model_manager.getSymbolClassifierModel(request.form['model'])
    except OSError as ex:
        return message('Error loading model. Specified model does not exist'), 404

    shape, position = model.predict(shape_image, position_image, n)
    
    if shape is None or position is None:
        return message('Error predicting symbol'), 404
    
    result = { 'shape': shape, 'position': position }
    return jsonify(result), 200


'''@app.route('/image/<id>/e2e', methods=['GET'])
def e2e_classify(id):
    if not _storage.exists(id):
        return message(f'Image [{id}] does not exist'), 404
    predictions = _e2e_classifier.predict(_storage.path(id))
    result = [{"shape": x[0].split(":")[0],
                "position": x[0].split(":")[1],
                "start": x[1],
                "end": x[2]} for x in predictions]
    return jsonify(result), 200
'''

@app.route('/image/<id>/e2e', methods=['POST'])
def e2e_classify(id):

    try:
        model = _model_manager.getE2EModel(request.form['model'])
    except IOError as e:
        return message('Error loading model. The requested model does not exist'), 404

    if not _storage.exists(id):
        return message(f'Image [{id}] does not exist'), 404
        
    # TO-DO SUBIR DAVID    
    try:
        left = int(request.form['left'])
        top = int(request.form['top'])
        right = int(request.form['right'])
        bottom = int(request.form['bottom'])
        n = int(request.form.get('predictions', "1"))
    except ValueError as e:
        return message('Wrong input values'), 400

    try:
        image = _storage.read(id, left, top, right, bottom)
    except Exception as e:
        return message('Error cropping image'), 400
    # END TO-DO SUBIR DAVID    
        
        
    predictions = model.predict(image)
    result = [{"shape": x[0].split(":")[0],
                "position": x[0].split(":")[1],
                "start": x[1],
                "end": x[2]} for x in predictions]
    return jsonify(result), 200

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Mensural symbol classification (predictor on the server).')
    
    # Symbol classification
    parser.add_argument('-sc_model_shape',    dest='sc_model_shape', type=str, default='model/symbol-classification/shape_classifier.h5')
    parser.add_argument('-sc_model_position', dest='sc_model_position',    type=str, default='model/symbol-classification/position_classifier.h5')
    parser.add_argument('-sc_vocabulary_shape',    dest='sc_vocabulary_shape', type=str, default='model/symbol-classification/symbol_shape_map.npy')
    parser.add_argument('-sc_vocabulary_position', dest='sc_vocabulary_position',    type=str, default='model/symbol-classification/symbol_position_map.npy')

    # End-to-end recognition
    parser.add_argument('-e2e_model', dest='e2e_model', type=str, default='model/end-to-end/model_muret_v0_40.meta')
    parser.add_argument('-e2e_vocabulary', dest='e2e_vocabulary', type=str, default='model/end-to-end/vocabulary.npy')

    # Server configuration
    parser.add_argument('-port', dest='port', type=int, default=8888)
    parser.add_argument('-image_storage', dest='image_storage', type=str, default='images')

    parser.add_argument('-ip', dest='ip', type=str, default='0.0.0.0')
    args = parser.parse_args()

    _model_manager = ModelManager(args.e2e_vocabulary, args.sc_vocabulary_shape, args.sc_vocabulary_position)

    # Initialize image storage
    _storage = ImageStorage(args.image_storage)

    # Create symbol classifier, which loads the models and the dictionary for the vocabularies
    #_symbol_classifier = SymbolClassifier(args.sc_model_shape, args.sc_model_position, args.sc_vocabulary_shape, args.sc_vocabulary_position)

    # Create end-to-end classifier
    #_e2e_classifier = E2EClassifier(args.e2e_model, args.e2e_vocabulary)


    # Start server, 0.0.0.0 allows connections from other computers
    app.run(host=args.ip, port=args.port)
