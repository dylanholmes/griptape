import pytest
from unittest.mock import Mock
from griptape.drivers import OpenSearchVectorStoreDriver
import numpy as np


class TestOpenSearchVectorStoreDriver:
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
        mock_client = mocker.patch(f'{target_module}.OpenSearch') \
            .return_value
        mock_client.index.return_value = { '_id': 'vector-id' }
        return mock_client
    
    @pytest.fixture
    def driver(self):
        return OpenSearchVectorStoreDriver(
            embedding_driver=Mock(),
            host='localhost',
            index_name='index-name',
        )

    def test_upsert_vector(self, driver, mock_client):
        # When
        vector_id = driver.upsert_vector([0.1, 0.2, 0.3], vector_id='vector-id', namespace='namespace', meta='meta')
        
        # Then
        mock_client.assert_called_once_with(
            index='index-name',
            id='vector-id',
            body={
                'vector': [0.1, 0.2, 0.3],
                'namespace': 'namespace',
                'metadata': 'meta'
            }
        )
        assert vector_id == 'vector-id'

    # def test_load_entry(self, driver):
    #     mock_entry = Mock()
    #     mock_entry.id = "foo2"
    #     mock_entry.vector = [2, 3, 4]
    #     mock_entry.meta = {"foo": "bar"}

    #     with patch.object(driver, "load_entry", return_value=mock_entry):
    #         entry = driver.load_entry("foo2", namespace="company")
    #         assert entry.id == "foo2"
    #         assert np.allclose(entry.vector, [2, 3, 4], atol=1e-6)
    #         assert entry.meta == {"foo": "bar"}

    # def test_load_entries(self, driver):
    #     mock_entry = Mock()
    #     mock_entry.id = "try_load"
    #     mock_entry.vector = [0.7, 0.8, 0.9]
    #     mock_entry.meta = None

    #     with patch.object(driver, "load_entries", return_value=[mock_entry]):
    #         entries = driver.load_entries(namespace="company")
    #         assert len(entries) == 1
    #         assert entries[0].id == "try_load"
    #         assert np.allclose(entries[0].vector, [0.7, 0.8, 0.9], atol=1e-6)
    #         assert entries[0].meta is None

    # def test_query(self, driver):
    #     mock_result = Mock()
    #     mock_result.id = "query_result"

    #     with patch.object(driver, "query", return_value=[mock_result]):
    #         query_string = "sample query text"
    #         results = driver.query(query_string, count=5, namespace="company")
    #         assert len(results) == 1, "Expected results from the query"
