import pytest
from unittest.mock import Mock
from griptape.drivers import AmazonOpenSearchVectorStoreDriver
import numpy as np



class TestAmazonOpenSearchVectorStoreDriver:
    @pytest.fixture(autouse=True)
    def mock_session(self, mocker):
        mock_session = mocker.patch('boto3.Session').return_value
        mock_session.get_credentials.return_value.access_key = 'access-key'
        mock_session.get_credentials.return_value.secret_key = 'secret-key'
        mock_session.region_name = 'region-name'
        return mock_session

    @pytest.fixture(autouse=True)
    def mock_http_auth(self, mocker):
        return mocker.patch('requests_aws4auth.AWS4Auth').return_value
    
    @pytest.fixture(autouse=True)
    def mock_client(self, mocker):
        target_module = 'griptape.drivers.vector.amazon_opensearch_vector_store_driver'
        mock_client = mocker.patch(f'{target_module}.OpenSearch').return_value
        mock_client.index.return_value = { '_id': 'vector-id' }
        mock_client.search.return_value = {
            'hits': {
                'total': { 'value': 1 },
                'hits': [{
                    '_source': {
                        'vector': [2, 3, 4],
                        'metadata': {'meta': 'data'},
                        'namespace': 'namespace',
                    },
                    '_score': 99
                }]
            }
        }
        return mock_client
    
    @pytest.fixture
    def driver(self, mock_session):
        return AmazonOpenSearchVectorStoreDriver(
            session=mock_session,
            embedding_driver=Mock(),
            host='localhost',
            index_name='index-name',
        )

    def test_upsert_vector(self, driver, mock_client):
        # When
        vector_id = driver.upsert_vector(
            vector=[0.1, 0.2, 0.3],
            vector_id='vector-id',
            namespace='namespace',
            meta='meta'
        )
        
        # Then
        mock_client.index.assert_called_once_with(
            index='index-name',
            id='vector-id',
            body={
                'vector': [0.1, 0.2, 0.3],
                'namespace': 'namespace',
                'metadata': 'meta'
            }
        )
        assert vector_id == 'vector-id'

    def test_load_entry(self, driver, mock_client):
        # When
        entry = driver.load_entry(vector_id='vector-id')

        # Then
        mock_client.search.assert_called_once_with(
            index='index-name',
            body={
                'query': {'bool': {'must': [{'term': {'_id':'vector-id'}}]}},
                'size': 1
            }
        )
        assert entry.id == 'vector-id'
        assert entry.vector == [2, 3, 4]
        assert entry.meta == {'meta': 'data'}
        assert entry.namespace == 'namespace'

    def test_load_entry_with_namespace(self, driver, mock_client):
        # When
        entry = driver.load_entry(vector_id='vector-id', namespace='namespace')

        # Then
        mock_client.search.assert_called_once_with(
            index='index-name',
            body={
                'query': {'bool': {'must': [
                    {'term': {'_id': 'vector-id'}},
                    {'term': {'namespace': 'namespace'}}
                ]}},
                'size': 1
            }
        )
        assert entry.id == 'vector-id'
        assert entry.vector == [2, 3, 4]
        assert entry.meta == {'meta': 'data'}
        assert entry.namespace == 'namespace'

    def test_load_entry_returns_none_when_entry_absent(self, driver, mock_client):
        # Given
        mock_client.search.return_value = {'hits': {'total': { 'value': 0 }}}

        # When
        entry = driver.load_entry(vector_id='vector-id')

        # Then
        assert entry is None

    def test_load_entry_returns_none_when_client_throws(self, driver, mock_client):
        # Given
        mock_client.search.side_effect = Exception

        # When
        entry = driver.load_entry(vector_id='vector-id')

        # Then
        assert entry is None

    def test_load_entries(self, driver, mock_client):
        # Given
        mock_client.search.return_value = {
            'hits': {
                'total': { 'value': 2 },
                'hits': [
                    {
                        '_id': 'vector-a',
                        '_source': {
                            'vector': [0xA, 1, 2],
                            'metadata': {'meta': 'a'},
                            'namespace': 'namespace-a',
                        }
                    },
                    {
                        '_id': 'vector-b',
                        '_source': {
                            'vector': [0XB, 1, 2],
                            'metadata': {'meta': 'b'},
                            'namespace': 'namespace-b',
                        }
                    }
                ]
            }
        }

        # When
        entries = driver.load_entries()

        # Then
        mock_client.search.assert_called_once_with(
            index='index-name',
            body={'query': {'match_all': {}}, 'size': 10000}
        )
        assert entries[0].id == 'vector-a'
        assert entries[0].vector == [0xA, 1, 2]
        assert entries[0].meta == {'meta': 'a'}
        assert entries[0].namespace == 'namespace-a'
        assert entries[1].id == 'vector-b'
        assert entries[1].vector == [0XB, 1, 2]
        assert entries[1].meta == {'meta': 'b'}
        assert entries[1].namespace == 'namespace-b'


    def test_load_entries_with_namespace(self, driver, mock_client):
        # Given
        mock_client.search.return_value = {
            'hits': {
                'total': { 'value': 2 },
                'hits': [
                    {
                        '_id': 'vector-a',
                        '_source': {
                            'vector': [0xA, 1, 2],
                            'metadata': {'meta': 'a'},
                            'namespace': 'namespace',
                        }
                    },
                    {
                        '_id': 'vector-b',
                        '_source': {
                            'vector': [0XB, 1, 2],
                            'metadata': {'meta': 'b'},
                            'namespace': 'namespace',
                        }
                    }
                ]
            }
        }

        # When
        entries = driver.load_entries(namespace='namespace')

        # Then
        mock_client.search.assert_called_once_with(
            index='index-name',
            body={'query': { 'match': { 'namespace': 'namespace' } }, 'size': 10000}
        )
        assert entries[0].id == 'vector-a'
        assert entries[0].vector == [0xA, 1, 2]
        assert entries[0].meta == {'meta': 'a'}
        assert entries[0].namespace == 'namespace'
        assert entries[1].id == 'vector-b'
        assert entries[1].vector == [0XB, 1, 2]
        assert entries[1].meta == {'meta': 'b'}
        assert entries[1].namespace == 'namespace'

    def test_query(self, driver, mock_client):
        # Given
        mock_client.search.return_value = {
            'hits': {
                'total': { 'value': 1 },
                'hits': [{
                    '_source': {
                        'vector': [2, 3, 4],
                        'metadata': {'meta': 'data'},
                        'namespace': 'namespace',
                    }
                }]
            }
        }

        # When
        results = driver.query([0, 1, 2])

        # Then
        mock_client.search.assert_called_once_with(
            index='index-name',
            body={'query': {'knn': {'vector': {'vector': [1, 2, 3], 'count': 5}}}, 'size': 5}
        )
        assert results == ['fuzz']


        # with patch.object(driver, "query", return_value=[mock_result]):
        #     query_vector = [0.5, 0.5, 0.5]
        #     results = driver.query(query_vector, count=5, namespace="company")
        #     assert len(results) == 1, "Expected results from the query"


    # def test_upsert_vector(self, mock_session):
    #     driver = AmazonOpenSearchVectorStoreDriver(session=mock_session)
    #     assert driver.upsert_vector([0.1, 0.2, 0.3], vector_id="foo", namespace="company") == "foo"
